"""Small, dependency-free helpers for safe JSON persistence.

The application still uses files for local deployments. These helpers prevent
partial writes and serialize read/modify/write operations across Gunicorn
threads and processes on Unix systems.
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, TypeVar

import fcntl

T = TypeVar("T")


@contextmanager
def exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_json(path: Path, default: T) -> T:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return default


def write_json_atomic(path: Path, value, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=indent, default=str)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as temp_file:
            temp_file.write(payload)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def update_json_atomic(path: Path, default: T, update: Callable[[T], T]) -> T:
    with exclusive_file_lock(path):
        current = read_json(path, default)
        updated = update(current)
        write_json_atomic(path, updated, indent=2)
        return updated
