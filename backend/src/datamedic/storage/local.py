"""Development storage adapter backed by a private local directory.

Keys are validated against a strict pattern before any filesystem access, so path traversal
is impossible. Objects are write-once: commit() refuses to replace an existing object.
"""

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from datamedic.storage.base import (
    InvalidObjectKey,
    ObjectAlreadyExists,
    ObjectNotFound,
    StoredObject,
)
from datamedic.storage.keys import KEY_RE, PREFIX_RE


class _LocalWriter:
    def __init__(self, final_path: Path, tmp_dir: Path, key: str) -> None:
        self._final = final_path
        self._key = key
        self._tmp = tmp_dir / f"{uuid.uuid4()}.part"
        self._fh: BinaryIO | None = open(self._tmp, "xb")  # noqa: SIM115
        self._hash = hashlib.sha256()
        self._size = 0

    def write(self, chunk: bytes) -> None:
        if self._fh is None:
            raise RuntimeError("writer is closed")
        self._fh.write(chunk)
        self._hash.update(chunk)
        self._size += len(chunk)

    def commit(self) -> StoredObject:
        if self._fh is None:
            raise RuntimeError("writer is closed")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None
        self._final.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Reserve the final name atomically; fails if the object already exists.
            fd = os.open(self._final, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError as exc:
            self._tmp.unlink(missing_ok=True)
            raise ObjectAlreadyExists(self._key) from exc
        os.replace(self._tmp, self._final)
        return StoredObject(self._key, self._size, self._hash.digest())

    def abort(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._tmp.unlink(missing_ok=True)


class LocalFilesystemStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._objects = self._root / "objects"
        self._tmp = self._root / "tmp"
        self._objects.mkdir(parents=True, exist_ok=True)
        self._tmp.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not KEY_RE.fullmatch(key):
            raise InvalidObjectKey("invalid object key")
        path = (self._objects / key).resolve()
        if not path.is_relative_to(self._objects):
            raise InvalidObjectKey("invalid object key")
        return path

    def _prefix_path(self, prefix: str) -> Path:
        if not PREFIX_RE.fullmatch(prefix):
            raise InvalidObjectKey("invalid prefix")
        path = (self._objects / prefix).resolve()
        if not path.is_relative_to(self._objects):
            raise InvalidObjectKey("invalid prefix")
        return path

    def open_writer(self, key: str) -> _LocalWriter:
        return _LocalWriter(self._path(key), self._tmp, key)

    def open_reader(self, key: str) -> BinaryIO:
        path = self._path(key)
        try:
            return open(path, "rb")  # noqa: SIM115
        except FileNotFoundError as exc:
            raise ObjectNotFound(key) from exc

    def size(self, key: str) -> int:
        try:
            return self._path(key).stat().st_size
        except FileNotFoundError as exc:
            raise ObjectNotFound(key) from exc

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_prefix(self, prefix: str) -> list[str]:
        base = self._prefix_path(prefix)
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self._objects).as_posix() for p in base.rglob("*") if p.is_file()
        )

    def modified_at(self, key: str) -> float:
        try:
            return self._path(key).stat().st_mtime
        except FileNotFoundError as exc:
            raise ObjectNotFound(key) from exc

    def delete(self, key: str) -> bool:
        if key.startswith("raw/"):
            raise InvalidObjectKey("originals are never deleted individually")
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def delete_prefix(self, prefix: str) -> int:
        base = self._prefix_path(prefix)
        if not base.exists():
            return 0
        count = sum(1 for p in base.rglob("*") if p.is_file())
        shutil.rmtree(base)
        return count

    def check_writable(self) -> bool:
        probe = self._tmp / f"probe-{uuid.uuid4()}"
        try:
            probe.write_bytes(b"")
            probe.unlink()
            return True
        except OSError:
            return False
