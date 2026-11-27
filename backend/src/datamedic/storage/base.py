"""Private object storage interface.

Deliberately has no URL-producing methods: in the MVP, browsers never receive storage URLs.
All reads and writes go through the API or workers after an ownership check.
Implementations: LocalFilesystemStorage (development); an S3 adapter is planned.
"""

from dataclasses import dataclass
from typing import BinaryIO, Protocol


class StorageError(Exception):
    pass


class InvalidObjectKey(StorageError):
    pass


class ObjectAlreadyExists(StorageError):
    pass


class ObjectNotFound(StorageError):
    pass


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int
    sha256: bytes


class ObjectWriter(Protocol):
    def write(self, chunk: bytes) -> None: ...

    def commit(self) -> StoredObject:
        """Finalise the object. Fails with ObjectAlreadyExists: objects are write-once."""
        ...

    def abort(self) -> None: ...


class ObjectStorage(Protocol):
    def open_writer(self, key: str) -> ObjectWriter: ...

    def open_reader(self, key: str) -> BinaryIO: ...

    def size(self, key: str) -> int: ...

    def exists(self, key: str) -> bool: ...

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under ``prefix``; returns the number deleted."""
        ...

    def list_prefix(self, prefix: str) -> list[str]: ...

    def modified_at(self, key: str) -> float:
        """POSIX timestamp of when the object was written."""
        ...

    def delete(self, key: str) -> bool:
        """Delete one derived object (never under ``raw/``). Used only by orphan
        reconciliation for objects no row references. Returns False if it did not exist."""
        ...

    def check_writable(self) -> bool: ...
