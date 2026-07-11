from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pyqwest import Client, SyncClient
from pyqwest.testing import ASGITransport, WSGITransport

from connectrpc._client_async import _consume_single_response
from connectrpc.errors import ConnectError

from .haberdasher_connect import (
    Haberdasher,
    HaberdasherASGIApplication,
    HaberdasherClient,
    HaberdasherClientSync,
    HaberdasherSync,
    HaberdasherWSGIApplication,
)
from .haberdasher_pb import Hat, Size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleHaberdasher(Haberdasher):
    async def make_hat(self, request, ctx):
        return Hat()


class _SimpleHaberdasherSync(HaberdasherSync):
    def make_hat(self, request, ctx):
        return Hat()


def _make_sync_client():
    transport = WSGITransport(HaberdasherWSGIApplication(_SimpleHaberdasherSync()))
    return HaberdasherClientSync(
        "http://localhost", http_client=SyncClient(transport=transport)
    )


def _make_async_client():
    transport = ASGITransport(HaberdasherASGIApplication(_SimpleHaberdasher()))
    return HaberdasherClient(
        "http://localhost", http_client=Client(transport=transport)
    )


# ---------------------------------------------------------------------------
# Bug 1: close() actually releases underlying HTTP client resources
# ---------------------------------------------------------------------------


class TestCloseReleasesResources:
    def test_sync_close_calls_underlying_close(self):
        mock_http = MagicMock()
        mock_http.close = MagicMock()
        client = HaberdasherClientSync("http://localhost", http_client=mock_http)

        client.close()

        mock_http.close.assert_called_once()
        assert client._closed is True

    def test_sync_close_idempotent(self):
        mock_http = MagicMock()
        mock_http.close = MagicMock()
        client = HaberdasherClientSync("http://localhost", http_client=mock_http)

        client.close()
        client.close()

        mock_http.close.assert_called_once()

    def test_sync_close_without_underlying_close(self):
        """If the HTTP client has no close method, close() should still work."""
        transport = WSGITransport(HaberdasherWSGIApplication(_SimpleHaberdasherSync()))
        client = HaberdasherClientSync(
            "http://localhost", http_client=SyncClient(transport=transport)
        )
        client.close()  # Should not raise
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_async_close_calls_underlying_aclose(self):
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        client = HaberdasherClient("http://localhost", http_client=mock_http)

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_async_close_idempotent(self):
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        client = HaberdasherClient("http://localhost", http_client=mock_http)

        await client.close()
        await client.close()

        mock_http.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_close_without_underlying_aclose(self):
        """If the HTTP client has no aclose method, close() should still work."""
        transport = ASGITransport(HaberdasherASGIApplication(_SimpleHaberdasher()))
        client = HaberdasherClient(
            "http://localhost", http_client=Client(transport=transport)
        )
        await client.close()  # Should not raise
        assert client._closed is True


# ---------------------------------------------------------------------------
# Bug 2: execute_* raises RuntimeError after close
# ---------------------------------------------------------------------------


class TestExecuteAfterClose:
    def test_sync_execute_unary_after_close(self):
        client = _make_sync_client()
        client.close()
        with pytest.raises(RuntimeError, match="Client is closed"):
            client.make_hat(Size(inches=10))

    def test_sync_context_manager_closes(self):
        client = _make_sync_client()
        with client:
            pass
        with pytest.raises(RuntimeError, match="Client is closed"):
            client.make_hat(Size(inches=10))

    @pytest.mark.asyncio
    async def test_async_execute_unary_after_close(self):
        client = _make_async_client()
        await client.close()
        with pytest.raises(RuntimeError, match="Client is closed"):
            await client.make_hat(Size(inches=10))

    @pytest.mark.asyncio
    async def test_async_context_manager_closes(self):
        client = _make_async_client()
        async with client:
            pass
        with pytest.raises(RuntimeError, match="Client is closed"):
            await client.make_hat(Size(inches=10))


# ---------------------------------------------------------------------------
# Bug 3: _consume_single_response closes the async generator
# ---------------------------------------------------------------------------


class TestConsumeSingleResponseClosesGenerator:
    @pytest.mark.asyncio
    async def test_closes_stream_on_success(self):
        """After single response, stream.aclose() is called."""
        aclose_called = False

        class MockStream:
            def __init__(self):
                self._consumed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._consumed:
                    self._consumed = True
                    return "msg"
                raise StopAsyncIteration

            async def aclose(self):
                nonlocal aclose_called
                aclose_called = True

        result = await _consume_single_response(MockStream())
        assert result == "msg"
        assert aclose_called

    @pytest.mark.asyncio
    async def test_closes_stream_on_multiple_messages(self):
        """Even on error, stream.aclose() is called."""
        aclose_called = False

        class MockStream:
            def __init__(self):
                self._count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                self._count += 1
                return f"msg{self._count}"

            async def aclose(self):
                nonlocal aclose_called
                aclose_called = True

        with pytest.raises(ConnectError, match="multiple messages"):
            await _consume_single_response(MockStream())
        assert aclose_called

    @pytest.mark.asyncio
    async def test_raises_on_empty_generator(self):
        async def gen():
            return
            yield  # Make it an async generator

        with pytest.raises(ConnectError, match="zero messages"):
            await _consume_single_response(gen())
