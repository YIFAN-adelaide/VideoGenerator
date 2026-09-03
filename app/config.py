from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "9100"))
    video_provider: str = os.getenv("VIDEO_PROVIDER", "mock")
    output_dir: str = os.getenv("OUTPUT_DIR", "./outputs")
    max_concurrent_generations: int = int(
        os.getenv("MAX_CONCURRENT_GENERATIONS", "1")
    )

    # External FastVideo serving runtime. The default model is our next
    # benchmark target; the same provider can later point at HunyuanVideo 1.5.
    fastvideo_base_url: str = os.getenv(
        "FASTVIDEO_BASE_URL", "http://127.0.0.1:9200"
    )
    fastvideo_model: str = os.getenv(
        "FASTVIDEO_MODEL",
        "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    )
    fastvideo_poll_interval_seconds: float = float(
        os.getenv("FASTVIDEO_POLL_INTERVAL_SECONDS", "1.0")
    )
    fastvideo_request_timeout_seconds: float = float(
        os.getenv("FASTVIDEO_REQUEST_TIMEOUT_SECONDS", "30.0")
    )
    fastvideo_job_timeout_seconds: float = float(
        os.getenv("FASTVIDEO_JOB_TIMEOUT_SECONDS", "1800.0")
    )
    # FastWan2.2 TI2V-5B is validated as a 720p-class model. 1280x704 is
    # Wan's native landscape shape. Keep both sizes configurable so the same
    # HTTP adapter can later serve a 480p Hunyuan checkpoint.
    fastvideo_480p_size: str = os.getenv(
        "FASTVIDEO_480P_SIZE", "832x480"
    )
    fastvideo_720p_size: str = os.getenv(
        "FASTVIDEO_720P_SIZE", "1280x704"
    )

    helios_repo_path: str | None = os.getenv("HELIOS_REPO_PATH")
    helios_base_model_path: str = os.getenv(
        "HELIOS_BASE_MODEL_PATH", "BestWishYsh/Helios-Distilled"
    )
    helios_transformer_path: str | None = os.getenv("HELIOS_TRANSFORMER_PATH")
    helios_device: str = os.getenv("HELIOS_DEVICE", "cuda:0")
    helios_weight_dtype: str = os.getenv("HELIOS_WEIGHT_DTYPE", "bf16")
    helios_low_vram: bool = _env_bool("HELIOS_LOW_VRAM", True)
    helios_group_offloading_type: str = os.getenv(
        "HELIOS_GROUP_OFFLOADING_TYPE", "leaf_level"
    )
    helios_num_blocks_per_group: int = int(
        os.getenv("HELIOS_NUM_BLOCKS_PER_GROUP", "4")
    )
    helios_enable_compile: bool = _env_bool("HELIOS_ENABLE_COMPILE", False)
    helios_disable_flash_attention: bool = _env_bool(
        "HELIOS_DISABLE_FLASH_ATTENTION", False
    )


settings = Settings()
