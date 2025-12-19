"""
Path helpers for locating project-level directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def _repo_root() -> Path:
    """Return the repository root (one level above the iris_demo package)."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: PathLike) -> Path:
    """Resolve a path relative to the repository root if it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _repo_root() / path


def resolve_sessions_directory(path_value: Optional[PathLike] = None) -> Path:
    """Resolve the sessions directory using the provided path or config default."""
    if path_value is None:
        from .llm import get_config

        path_value = get_config().sessions_directory
    return resolve_repo_path(path_value)
