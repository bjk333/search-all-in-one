"""mcp_http: 零新增依赖的原生 asyncio MCP Streamable HTTP 常驻共享服务端（默认仅绑定本机回环）。"""

from __future__ import annotations

import asyncio
import hmac
import json
from typing import Any
from urllib.parse import urlsplit
import uuid

from omni_search import __version__
from omni_search.mcp_server import OmniMCPServer

_MAX_BODY_BYTES = 2 * 1024 * 1024  # 请求体硬上限 2 MiB
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_STATUS_TEXT = {
    200: "OK",
    202: "Accepted",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    500: "Internal Server Error",
}


def _is_loopback_host(host: str) -> bool:
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


def _is_allowed_origin(origin: str) -> bool:
    """防 DNS Rebinding / 跨站请求伪造：若存在 Origin 头，仅放行本机回环源。"""
    if not origin:
        return True
    try:
        p = urlsplit(origin.strip())
        if p.scheme.lower() not in ("http", "https"):
            return False
        return _is_loopback_host(p.hostname or "")
    except Exception:
        return False


class McpHttpServer:
    """原生 asyncio MCP Streamable HTTP 服务端（支持 JSON 响应与 SSE notifications/progress 流）。"""

    def __init__(
        self,
        mcp_server: OmniMCPServer,
        host: str = "127.0.0.1",
        port: int = 8765,
        auth_token: str = "",
    ):
        eff_host = (host or "127.0.0.1").strip()
        eff_token = (auth_token or "").strip()
        if not _is_loopback_host(eff_host) and not eff_token:
            raise ValueError(
                f"refusing to bind non-loopback host '{eff_host}' without auth_token / OMNI_MCP_TOKEN"
            )
        self.mcp_server = mcp_server
        self.host = eff_host
        self.requested_port = int(port)
        self.port: int = int(port)
        self.auth_token = eff_token
        self.session_id = uuid.uuid4().hex
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_conn, self.host, self.requested_port)
        socks = self._server.sockets or []
        if socks:
            self.port = int(socks[0].getsockname()[1])

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        try:
            async with self._server:
                await self._server.serve_forever()
        finally:
            await self.stop()

    @staticmethod
    async def _send_response(
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes = b"",
        content_type: str = "application/json; charset=utf-8",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        reason = _STATUS_TEXT.get(status, "OK")
        hdrs = [
            f"HTTP/1.1 {status} {reason}",
            f"Content-Length: {len(body)}",
            "Connection: close",
            "MCP-Protocol-Version: 2025-06-18",
        ]
        if body or status == 200:
            hdrs.append(f"Content-Type: {content_type}")
        if extra_headers:
            for k, v in extra_headers.items():
                hdrs.append(f"{k}: {v}")
        raw = ("\r\n".join(hdrs) + "\r\n\r\n").encode("latin-1") + body
        writer.write(raw)
        await writer.drain()

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req_line_raw = await asyncio.wait_for(reader.readline(), timeout=15.0)
            if not req_line_raw:
                return
            parts = req_line_raw.decode("latin-1", errors="replace").strip().split()
            if len(parts) < 2:
                await self._send_response(writer, 400, b'{"error":"bad request line"}')
                return
            method, raw_target = parts[0].upper(), parts[1]
            path = urlsplit(raw_target).path

            headers: dict[str, str] = {}
            while True:
                hline = await asyncio.wait_for(reader.readline(), timeout=15.0)
                if not hline or hline in (b"\r\n", b"\n"):
                    break
                decoded = hline.decode("latin-1", errors="replace").strip()
                if ":" in decoded:
                    k, v = decoded.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_len = 0
            if "content-length" in headers:
                try:
                    content_len = max(0, int(headers["content-length"]))
                except ValueError:
                    await self._send_response(writer, 400, b'{"error":"invalid content-length"}')
                    return
            if content_len > _MAX_BODY_BYTES:
                await self._send_response(writer, 413, b'{"error":"payload too large"}')
                return
            body = await asyncio.wait_for(reader.readexactly(content_len), timeout=15.0) if content_len > 0 else b""

            # /health 公开轻量探活
            if path == "/health" and method == "GET":
                payload = json.dumps(
                    {
                        "ok": True,
                        "server": "search-all-in-one",
                        "version": __version__,
                        "transport": "streamable-http",
                        "engines": len(self.mcp_server.engines),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                await self._send_response(writer, 200, payload)
                return

            if path != "/mcp":
                await self._send_response(writer, 404, b'{"error":"not found"}')
                return

            # Origin 防 DNS Rebinding 校验
            origin = headers.get("origin", "")
            if origin and not _is_allowed_origin(origin):
                await self._send_response(writer, 403, b'{"error":"forbidden origin"}')
                return

            # 可选 Bearer Token 鉴权（常量时间比对）
            if self.auth_token:
                auth_hdr = headers.get("authorization", "")
                expected = f"Bearer {self.auth_token}"
                if not hmac.compare_digest(auth_hdr, expected):
                    await self._send_response(
                        writer,
                        401,
                        b'{"error":"unauthorized"}',
                        extra_headers={"WWW-Authenticate": "Bearer"},
                    )
                    return

            sess_hdr = {"Mcp-Session-Id": headers.get("mcp-session-id") or self.session_id}

            if method == "DELETE":
                await self._send_response(writer, 204, b"", extra_headers=sess_hdr)
                return

            if method != "POST":
                await self._send_response(writer, 405, b'{"error":"method not allowed"}', extra_headers=sess_hdr)
                return

            try:
                req_obj = json.loads(body.decode("utf-8"))
            except Exception as exc:
                err = json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}).encode("utf-8")
                await self._send_response(writer, 400, err, extra_headers=sess_hdr)
                return

            if not isinstance(req_obj, dict):
                err = json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "expected JSON-RPC object"}}).encode("utf-8")
                await self._send_response(writer, 400, err, extra_headers=sess_hdr)
                return

            accept = headers.get("accept", "").lower()
            wants_sse = "text/event-stream" in accept and req_obj.get("method") == "tools/call"

            if wants_sse:
                sse_events: list[dict[str, Any]] = []
                resp = await self.mcp_server.handle_request(req_obj, emit_notification=sse_events.append)
                if resp is None:
                    await self._send_response(writer, 202, b"", extra_headers=sess_hdr)
                    return
                chunks: list[str] = []
                for ev in sse_events:
                    chunks.append(f"event: message\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n")
                chunks.append(f"event: message\ndata: {json.dumps(resp, ensure_ascii=False)}\n\n")
                await self._send_response(
                    writer,
                    200,
                    "".join(chunks).encode("utf-8"),
                    content_type="text/event-stream; charset=utf-8",
                    extra_headers={**sess_hdr, "Cache-Control": "no-cache"},
                )
                return

            resp = await self.mcp_server.handle_request(req_obj)
            if resp is None:
                await self._send_response(writer, 202, b"", extra_headers=sess_hdr)
                return
            await self._send_response(
                writer,
                200,
                json.dumps(resp, ensure_ascii=False).encode("utf-8"),
                extra_headers=sess_hdr,
            )
        except Exception:
            try:
                await self._send_response(writer, 500, b'{"error":"internal server error"}')
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
