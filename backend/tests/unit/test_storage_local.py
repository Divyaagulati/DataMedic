import hashlib
import uuid
from pathlib import Path

import pytest

from datamedic.storage import (
    InvalidObjectKey,
    LocalFilesystemStorage,
    ObjectAlreadyExists,
    ObjectNotFound,
    dataset_prefix,
    raw_object_key,
)


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(tmp_path)


def test_write_read_and_hash(storage: LocalFilesystemStorage) -> None:
    key = raw_object_key(uuid.uuid4(), uuid.uuid4())
    w = storage.open_writer(key)
    w.write(b"hello ")
    w.write(b"world")
    obj = w.commit()
    assert obj.size_bytes == 11
    assert obj.sha256 == hashlib.sha256(b"hello world").digest()
    with storage.open_reader(key) as fh:
        assert fh.read() == b"hello world"


def test_objects_are_write_once(storage: LocalFilesystemStorage) -> None:
    key = raw_object_key(uuid.uuid4(), uuid.uuid4())
    w = storage.open_writer(key)
    w.write(b"original")
    w.commit()
    w2 = storage.open_writer(key)
    w2.write(b"tampered")
    with pytest.raises(ObjectAlreadyExists):
        w2.commit()
    with storage.open_reader(key) as fh:
        assert fh.read() == b"original"


def test_abort_leaves_nothing(storage: LocalFilesystemStorage, tmp_path: Path) -> None:
    key = raw_object_key(uuid.uuid4(), uuid.uuid4())
    w = storage.open_writer(key)
    w.write(b"partial")
    w.abort()
    assert not storage.exists(key)
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.parametrize(
    "key",
    [
        "../etc/passwd",
        "raw/../../x",
        "raw/not-a-uuid/also-not/x",
        f"raw/{uuid.uuid4()}/{uuid.uuid4()}/../../../x",
        f"secret/{uuid.uuid4()}/{uuid.uuid4()}/x",
        f"raw/{uuid.uuid4()}/{uuid.uuid4()}/My File.csv",
        "C:\\Windows\\win.ini",
    ],
)
def test_invalid_keys_rejected(storage: LocalFilesystemStorage, key: str) -> None:
    with pytest.raises(InvalidObjectKey):
        storage.open_writer(key)
    with pytest.raises(InvalidObjectKey):
        storage.open_reader(key)


def test_delete_prefix_scoped_to_dataset(storage: LocalFilesystemStorage) -> None:
    owner, ds_a, ds_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    for ds in (ds_a, ds_b):
        w = storage.open_writer(raw_object_key(owner, ds))
        w.write(b"x")
        w.commit()
    assert storage.delete_prefix(dataset_prefix("raw", owner, ds_a)) == 1
    assert storage.list_prefix(dataset_prefix("raw", owner, ds_a)) == []
    assert len(storage.list_prefix(dataset_prefix("raw", owner, ds_b))) == 1


def test_missing_object(storage: LocalFilesystemStorage) -> None:
    with pytest.raises(ObjectNotFound):
        storage.open_reader(raw_object_key(uuid.uuid4(), uuid.uuid4()))


def test_storage_has_no_url_methods() -> None:
    public = {n for n in dir(LocalFilesystemStorage) if not n.startswith("_")}
    assert not any("url" in n.lower() for n in public)
