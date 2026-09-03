from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from app.providers.base import ProviderResult, VideoProvider
from app.schemas import VideoGenerationRequest


class FastVideoProviderError(RuntimeError):
    """Raised when the external FastVideo runtime cannot complete a job."""


class FastVideoProvider(VideoProvider):
    """
    HTTP adapter for a long-lived FastVideo serving process.

    FastVideo owns the heavyweight model lifecycle, GPU memory, and its own
    serialized generation engine. This provider deliberately does *not* import
    torch, diffusers, or FastVideo itself.
    """

    TERMINAL_STATUSES = {"completed", "failed"}

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        output_dir: str,
        poll_interval_seconds: float = 1.0,
        request_timeout_seconds: float = 30.0,
        job_timeout_seconds: float = 1800.0,
        size_480p: str = "832x480",
        size_720p: str = "1280x704",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be >= 0")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        if job_timeout_seconds <= 0:
            raise ValueError("job_timeout_seconds must be > 0")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval_seconds = poll_interval_seconds
        self.job_timeout_seconds = job_timeout_seconds
        self.size_by_resolution = {
            "480p": size_480p,
            "720p": size_720p,
        }

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(request_timeout_seconds),
        )

    def build_payload(self, request: VideoGenerationRequest) -> dict[str, Any]:
        """
        Translate our stable application schema to FastVideo's OpenAI-style
        /v1/videos contract.

        We intentionally leave model-specific denoising steps and guidance
        unset. FastVideo's served-model preset should own those defaults (for
        example the 3-step schedule of distilled FastWan checkpoints).
        """
        try:
            size = self.size_by_resolution[request.resolution]
        except KeyError as exc:  # defensive; Pydantic currently prevents this.
            raise ValueError(
                f"Unsupported resolution for FastVideo: {request.resolution!r}"
            ) from exc

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": request.prompt,
            "seconds": request.duration_seconds,
            "fps": request.fps,
            "size": size,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    async def generate(
        self,
        request: VideoGenerationRequest,
        job_id: str,
    ) -> ProviderResult:
        payload = self.build_payload(request)
        submitted_at = time.perf_counter()

        upstream_job = await self._request_json(
            "POST",
            "/v1/videos",
            json=payload,
        )
        upstream_job_id = self._job_id(upstream_job)

        final_job = await self._wait_for_completion(
            upstream_job_id,
            initial_job=upstream_job,
        )

        output_path = self.output_dir / f"{job_id}.mp4"
        await self._download_video(upstream_job_id, output_path)

        total_seconds = time.perf_counter() - submitted_at
        metadata: dict[str, Any] = {
            "provider": "fastvideo",
            "runtime": "fastvideo_server",
            "model": self.model,
            "upstream_job_id": upstream_job_id,
            "upstream_status": final_job.get("status", "completed"),
            "requested_duration_seconds": request.duration_seconds,
            "fps": request.fps,
            "resolution": request.resolution,
            "size": payload["size"],
            "provider_total_seconds": total_seconds,
        }

        # Preserve useful server-side benchmark data without depending on one
        # exact FastVideo response version.
        for key in (
            "file_name",
            "file_path",
            "timings",
            "peak_memory_mb",
            "peak_memory",
            "inference_time_s",
            "created_at",
            "completed_at",
        ):
            if key in final_job:
                metadata[key] = final_job[key]

        return ProviderResult(
            output_path=str(output_path.resolve()),
            metadata=metadata,
        )

    async def health(self) -> dict[str, Any] | None:
        try:
            response = await self._client.get("/health")
            self._raise_for_status(response)
        except Exception as exc:
            return {
                "status": "unavailable",
                "runtime": "fastvideo_server",
                "base_url": self.base_url,
                "model": self.model,
                "error": f"{type(exc).__name__}: {exc}",
            }

        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}

        return {
            "status": "ok",
            "runtime": "fastvideo_server",
            "base_url": self.base_url,
            "model": self.model,
            "upstream": body,
        }

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _wait_for_completion(
        self,
        upstream_job_id: str,
        *,
        initial_job: dict[str, Any],
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.job_timeout_seconds
        job = initial_job

        while True:
            status = str(job.get("status", "")).lower()

            if status == "completed":
                return job

            if status == "failed":
                error = job.get("error") or job.get("message") or "unknown error"
                raise FastVideoProviderError(
                    f"FastVideo job {upstream_job_id} failed: {error}"
                )

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "FastVideo generation timed out after "
                    f"{self.job_timeout_seconds:.1f}s "
                    f"(upstream job {upstream_job_id})"
                )

            await asyncio.sleep(self.poll_interval_seconds)
            job = await self._request_json(
                "GET",
                f"/v1/videos/{upstream_job_id}",
            )

    async def _download_video(
        self,
        upstream_job_id: str,
        output_path: Path,
    ) -> None:
        temp_path = output_path.with_suffix(output_path.suffix + ".part")
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with self._client.stream(
                "GET",
                f"/v1/videos/{upstream_job_id}/content",
            ) as response:
                self._raise_for_status(response)
                with temp_path.open("wb") as file_obj:
                    async for chunk in response.aiter_bytes():
                        file_obj.write(chunk)
            temp_path.replace(output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    async def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise FastVideoProviderError(
                f"FastVideo transport error for {method} {path}: {exc}"
            ) from exc

        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            raise FastVideoProviderError(
                f"FastVideo returned non-JSON data for {method} {path}"
            ) from exc

        if not isinstance(body, dict):
            raise FastVideoProviderError(
                f"FastVideo returned an unexpected response for {method} {path}"
            )
        return body

    @staticmethod
    def _job_id(job: dict[str, Any]) -> str:
        value = job.get("id") or job.get("video_id") or job.get("job_id")
        if not value:
            raise FastVideoProviderError(
                "FastVideo submission response did not include a video id"
            )
        return str(value)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return

        message = response.text
        try:
            body = response.json()
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or message)
                elif error:
                    message = str(error)
                elif body.get("detail"):
                    message = str(body["detail"])
        except ValueError:
            pass

        raise FastVideoProviderError(
            f"FastVideo HTTP {response.status_code}: {message}"
        )
