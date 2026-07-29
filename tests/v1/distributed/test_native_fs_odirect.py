# SPDX-License-Identifier: Apache-2.0
"""O_DIRECT behavior of the compiled native FS connector."""

# Standard
import select
import time

# Third Party
import pytest

lmcache_fs = pytest.importorskip("lmcache.lmcache_fs")


def _wait_for_completion(client, timeout: float = 5.0):
    poller = select.poll()
    poller.register(client.event_fd(), select.POLLIN)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if poller.poll(50):
            completions = client.drain_completions()
            if completions:
                return completions[0]
    raise AssertionError("native connector completion timed out")


def test_odirect_falls_back_for_misaligned_buffer_address(tmp_path):
    client = lmcache_fs.LMCacheFSClient(str(tmp_path), 1, "", True)
    try:
        payload_storage = bytearray(4097)
        payload = memoryview(payload_storage)[1:]
        payload[:] = b"x" * len(payload)
        key = "model@00000000@01"

        store_id = client.submit_batch_set([key], [payload])
        completed_id, ok, error, _results = _wait_for_completion(client)
        assert completed_id == store_id
        assert ok, error

        loaded_storage = bytearray(4097)
        loaded = memoryview(loaded_storage)[1:]
        load_id = client.submit_batch_get([key], [loaded])
        completed_id, ok, error, results = _wait_for_completion(client)
        assert completed_id == load_id
        assert ok, error
        assert results == [True]
        assert loaded == payload
    finally:
        client.close()
