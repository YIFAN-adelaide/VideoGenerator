from __future__ import annotations

from abc import ABC, abstractmethod

from app.director.video_plan import DirectorRequest, VideoPlan


class DirectorError(RuntimeError):
    """Base exception for Director failures."""


class DirectorConfigurationError(DirectorError):
    """Raised when a Director is incorrectly configured."""


class DirectorPlanningError(DirectorError):
    """Raised when a Director cannot create a valid video plan."""


class BaseDirector(ABC):
    def __init__(self, *, director_name: str, model_name: str) -> None:
        if not director_name.strip():
            raise DirectorConfigurationError("director_name cannot be empty.")
        if not model_name.strip():
            raise DirectorConfigurationError("model_name cannot be empty.")
        self._director_name = director_name
        self._model_name = model_name

    @property
    def director_name(self) -> str:
        return self._director_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @abstractmethod
    async def create_plan(self, request: DirectorRequest) -> VideoPlan:
        raise NotImplementedError
