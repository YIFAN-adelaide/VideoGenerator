from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, Literal


WeightDType = Literal["bf16", "fp16", "fp32"]
GroupOffloadType = Literal["leaf_level", "block_level"]


class HeliosRuntimeError(RuntimeError):
    """Base error for the Helios runtime layer."""


class HeliosDependencyError(HeliosRuntimeError):
    """Raised when Helios/PyTorch/Diffusers dependencies are unavailable."""


class HeliosLoadError(HeliosRuntimeError):
    """Raised when the model pipeline cannot be loaded."""


class HeliosRuntimeNotLoadedError(HeliosRuntimeError):
    """Raised when the pipeline is requested before explicit startup loading."""


@dataclass(frozen=True)
class HeliosRuntimeConfig:
    base_model_path: str = "BestWishYsh/Helios-Distilled"
    transformer_path: str | None = None
    repo_path: str | None = None

    device: str = "cuda:0"
    weight_dtype: WeightDType = "bf16"

    low_vram: bool = True
    group_offloading_type: GroupOffloadType = "leaf_level"
    num_blocks_per_group: int = 4

    enable_compile: bool = False
    disable_flash_attention: bool = False
    hf_parallel_loading_workers: int = 8

    def __post_init__(self) -> None:
        if not self.base_model_path.strip():
            raise ValueError("base_model_path cannot be empty")
        if self.weight_dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"Unsupported weight_dtype: {self.weight_dtype}")
        if self.group_offloading_type not in {"leaf_level", "block_level"}:
            raise ValueError("group_offloading_type must be leaf_level or block_level")
        if self.num_blocks_per_group < 1:
            raise ValueError("num_blocks_per_group must be >= 1")
        if self.enable_compile and self.low_vram:
            raise ValueError(
                "Helios compile mode and low-VRAM group offloading cannot be enabled together."
            )
        if not self.device.startswith("cuda"):
            raise ValueError("Helios runtime is GPU-only; use a CUDA device such as cuda:0.")
        if self.hf_parallel_loading_workers < 1:
            raise ValueError("hf_parallel_loading_workers must be >= 1")

    @property
    def resolved_transformer_path(self) -> str:
        return self.transformer_path or self.base_model_path


class HeliosModelLoader:
    """Load one Helios pipeline once and keep it resident for reuse."""

    def __init__(self, config: HeliosRuntimeConfig) -> None:
        self.config = config
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._attention_backend: str | None = None
        self._loaded_at: str | None = None
        self._load_seconds: float | None = None
        self._lock = RLock()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def get_pipeline(self) -> Any:
        if self._pipeline is None:
            raise HeliosRuntimeNotLoadedError(
                "Helios is not loaded. Load it during application startup first."
            )
        return self._pipeline

    def load(self) -> Any:
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            started = time.perf_counter()
            self._prepare_environment()
            runtime = self._import_runtime()
            torch = runtime.torch

            device = self._prepare_cuda(torch)
            weight_dtype = self._resolve_dtype(torch)

            try:
                transformer = runtime.HeliosTransformer3DModel.from_pretrained(
                    self.config.resolved_transformer_path,
                    subfolder="transformer",
                    torch_dtype=weight_dtype,
                )

                if not self.config.enable_compile:
                    transformer = runtime.replace_rmsnorm_with_fp32(transformer)
                    transformer = runtime.replace_all_norms_with_flash_norms(transformer)
                    runtime.replace_rope_with_flash_rope()

                self._attention_backend = self._configure_attention(
                    torch, transformer, device
                )

                vae = runtime.AutoencoderKLWan.from_pretrained(
                    self.config.base_model_path,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                )

                scheduler = runtime.HeliosScheduler.from_pretrained(
                    self.config.base_model_path,
                    subfolder="scheduler",
                )

                pipe = runtime.HeliosPipeline.from_pretrained(
                    self.config.base_model_path,
                    transformer=transformer,
                    vae=vae,
                    scheduler=scheduler,
                    torch_dtype=weight_dtype,
                )

                if self.config.enable_compile:
                    torch.backends.cudnn.benchmark = True
                    pipe.text_encoder.compile(
                        mode="max-autotune-no-cudagraphs", dynamic=False
                    )
                    pipe.vae.compile(
                        mode="max-autotune-no-cudagraphs", dynamic=False
                    )
                    pipe.transformer.compile(
                        mode="max-autotune-no-cudagraphs", dynamic=False
                    )

                if self.config.low_vram:
                    pipe.enable_group_offload(
                        onload_device=device,
                        offload_device=torch.device("cpu"),
                        offload_type=self.config.group_offloading_type,
                        num_blocks_per_group=(
                            self.config.num_blocks_per_group
                            if self.config.group_offloading_type == "block_level"
                            else None
                        ),
                        use_stream=True,
                        record_stream=True,
                    )
                else:
                    pipe = pipe.to(device)

            except Exception as exc:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise HeliosLoadError(
                    f"Failed to load Helios from {self.config.base_model_path!r}: {exc}"
                ) from exc

            self._pipeline = pipe
            self._torch = torch
            self._device = device
            self._loaded_at = datetime.now(timezone.utc).isoformat()
            self._load_seconds = time.perf_counter() - started
            return pipe

    def unload(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return

            torch = self._torch
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(self._device)
                except Exception:
                    pass

            self._pipeline = None
            self._device = None
            self._attention_backend = None
            self._loaded_at = None
            self._load_seconds = None

            gc.collect()

            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

            self._torch = None

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "loaded": self.is_loaded,
            "base_model_path": self.config.base_model_path,
            "transformer_path": self.config.resolved_transformer_path,
            "device": self.config.device,
            "weight_dtype": self.config.weight_dtype,
            "low_vram": self.config.low_vram,
            "group_offloading_type": (
                self.config.group_offloading_type if self.config.low_vram else None
            ),
            "attention_backend": self._attention_backend,
            "loaded_at": self._loaded_at,
            "load_seconds": self._load_seconds,
        }

        if not self.is_loaded or self._torch is None:
            return info

        torch = self._torch
        if torch.cuda.is_available():
            index = self._device.index if self._device.index is not None else 0
            props = torch.cuda.get_device_properties(index)
            info["cuda"] = {
                "device_name": torch.cuda.get_device_name(index),
                "compute_capability": ".".join(
                    str(x) for x in torch.cuda.get_device_capability(index)
                ),
                "total_vram_gib": round(props.total_memory / (1024**3), 2),
                "allocated_vram_gib": round(
                    torch.cuda.memory_allocated(index) / (1024**3), 2
                ),
                "reserved_vram_gib": round(
                    torch.cuda.memory_reserved(index) / (1024**3), 2
                ),
            }
        return info

    def _prepare_environment(self) -> None:
        os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "yes")
        os.environ.setdefault(
            "HF_PARALLEL_LOADING_WORKERS",
            str(self.config.hf_parallel_loading_workers),
        )

        if self.config.repo_path:
            repo = Path(self.config.repo_path).expanduser().resolve()
            if not repo.exists():
                raise HeliosLoadError(f"HELIOS_REPO_PATH does not exist: {repo}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))

    @staticmethod
    def _import_runtime() -> SimpleNamespace:
        try:
            import torch
            from diffusers.models import AutoencoderKLWan
            from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
            from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
            from helios.diffusers_version.transformer_helios_diffusers import (
                HeliosTransformer3DModel,
            )
            from helios.modules.helios_kernels import (
                replace_all_norms_with_flash_norms,
                replace_rmsnorm_with_fp32,
                replace_rope_with_flash_rope,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise HeliosDependencyError(
                "Helios runtime dependencies are unavailable. This is expected "
                "on the local mock machine. Install the official Helios environment "
                "on AWS and set HELIOS_REPO_PATH to the cloned repository."
            ) from exc

        return SimpleNamespace(
            torch=torch,
            AutoencoderKLWan=AutoencoderKLWan,
            HeliosPipeline=HeliosPipeline,
            HeliosScheduler=HeliosScheduler,
            HeliosTransformer3DModel=HeliosTransformer3DModel,
            replace_all_norms_with_flash_norms=replace_all_norms_with_flash_norms,
            replace_rmsnorm_with_fp32=replace_rmsnorm_with_fp32,
            replace_rope_with_flash_rope=replace_rope_with_flash_rope,
        )

    def _prepare_cuda(self, torch: Any) -> Any:
        if not torch.cuda.is_available():
            raise HeliosLoadError(
                "CUDA is unavailable. Refusing to silently fall back to CPU."
            )

        device = torch.device(self.config.device)
        index = device.index if device.index is not None else 0

        if index >= torch.cuda.device_count():
            raise HeliosLoadError(
                f"Requested {self.config.device}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )

        torch.cuda.set_device(index)

        if (
            self.config.weight_dtype == "bf16"
            and hasattr(torch.cuda, "is_bf16_supported")
            and not torch.cuda.is_bf16_supported()
        ):
            raise HeliosLoadError(
                "BF16 was requested but the GPU does not report BF16 support."
            )

        return device

    def _resolve_dtype(self, torch: Any) -> Any:
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.config.weight_dtype]

    def _configure_attention(self, torch: Any, transformer: Any, device: Any) -> str:
        if self.config.disable_flash_attention:
            return "disabled"

        major, _ = torch.cuda.get_device_capability(device)

        if major >= 9:
            try:
                transformer.set_attention_backend("_flash_3_hub")
                return "_flash_3_hub"
            except Exception:
                transformer.set_attention_backend("flash_hub")
                return "flash_hub"

        transformer.set_attention_backend("flash_hub")
        return "flash_hub"
