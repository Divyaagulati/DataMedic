from types import SimpleNamespace

from datamedic.workers.limits import MIB, apply_worker_memory_limit


def test_zero_means_no_limit() -> None:
    calls: list[object] = []
    fake = SimpleNamespace(RLIMIT_DATA=2, setrlimit=lambda *a: calls.append(a))
    assert apply_worker_memory_limit(0, fake) is False and calls == []


def test_limit_is_applied_as_rlimit_data() -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    fake = SimpleNamespace(RLIMIT_DATA=2, setrlimit=lambda res, lim: calls.append((res, lim)))
    assert apply_worker_memory_limit(2000, fake) is True
    assert calls == [(2, (2000 * MIB, 2000 * MIB))]
