"""Tests for KernelBusSubscriber endpoint resolution (Codex review #4 / Fix 3).

The subscriber used to hard-code ``http://localhost:6931/v1/events/stream`` as
its default. In any non-default kernel topology that meant sends and receives
targeted different hosts. The fix: resolve the URL from ``COGOS_ENDPOINT`` (with
the same default as the rest of the cogos client code) at construction time.

Run: python -m pytest tests/test_bus_bridge.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bus_bridge  # noqa: E402
from bus_bridge import KernelBusSubscriber, default_stream_url  # noqa: E402


def test_default_stream_url_uses_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("COGOS_ENDPOINT", raising=False)
    assert default_stream_url() == "http://localhost:6931/v1/events/stream"


def test_default_stream_url_honors_cogos_endpoint(monkeypatch):
    monkeypatch.setenv("COGOS_ENDPOINT", "http://kernel.internal:7000")
    assert default_stream_url() == "http://kernel.internal:7000/v1/events/stream"


def test_default_stream_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("COGOS_ENDPOINT", "http://kernel.internal:7000/")
    assert default_stream_url() == "http://kernel.internal:7000/v1/events/stream"


def test_subscriber_default_url_uses_env(monkeypatch):
    """A KernelBusSubscriber constructed with no url must honor COGOS_ENDPOINT."""
    monkeypatch.setenv("COGOS_ENDPOINT", "http://kernel.internal:7777")
    sub = KernelBusSubscriber()
    assert sub._url == "http://kernel.internal:7777/v1/events/stream"


def test_subscriber_explicit_url_overrides_env(monkeypatch):
    """An explicit url= argument must beat the env var (back-compat)."""
    monkeypatch.setenv("COGOS_ENDPOINT", "http://kernel.internal:7777")
    sub = KernelBusSubscriber(url="http://override.example/v1/events/stream")
    assert sub._url == "http://override.example/v1/events/stream"


def test_kernel_bus_stream_url_module_attr_resolves_default(monkeypatch):
    """The module-level back-compat attribute reflects the env at import.

    We can't reload the module mid-test, but we can verify it agrees with
    default_stream_url() called from the same env state.
    """
    # Just confirm the back-compat attr exists and is a string of expected shape
    assert isinstance(bus_bridge.KERNEL_BUS_STREAM_URL, str)
    assert bus_bridge.KERNEL_BUS_STREAM_URL.endswith("/v1/events/stream")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
