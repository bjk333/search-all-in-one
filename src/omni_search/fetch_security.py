"""fetch_security: SSRF 白名单拦截、混淆主机校验与受控逐跳流式字节读取。"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

_MAX_BYTES = 2 * 1024 * 1024  # 普通网页/文本 2 MiB 硬上限
_MAX_PDF_BYTES = 6 * 1024 * 1024  # PDF 文献 6 MiB 硬上限
_MAX_PDF_PAGES = 30
_MAX_REDIRECTS = 5

_OCTAL_OR_HEX_PART_RE = re.compile(r"^(?:0[0-7]+|0x[0-9a-f]+)$", re.IGNORECASE)


def _is_forbidden_ip(ip_str: str) -> bool:
    """严格白名单语义：仅放行公网全局路由 IP（自动解包 IPv4-mapped / 6to4 / Teredo，阻断 100.64.0.0/10 含阿里云元数据）。"""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True

    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None and _is_forbidden_ip(str(ip.ipv4_mapped)):
            return True
        if ip.sixtofour is not None and _is_forbidden_ip(str(ip.sixtofour)):
            return True
        if ip.teredo is not None:
            for inner in ip.teredo:
                if _is_forbidden_ip(str(inner)):
                    return True

    return (
        (not ip.is_global)
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_obfuscated_ip_host(host: str) -> bool:
    """识别纯十进制整数 IP、十六进制 IP 或带前导零八进制 IPv4（如 0177.0.0.1 / 2130706433 / 0x7f000001）。"""
    if not host:
        return False
    if host.isdigit():
        return True
    if host.startswith("0x") and all(c in "0123456789abcdef" for c in host[2:]):
        return True
    parts = host.split(".")
    if 2 <= len(parts) <= 4 and all(p.isdigit() or _OCTAL_OR_HEX_PART_RE.match(p) for p in parts):
        if any(_OCTAL_OR_HEX_PART_RE.match(p) for p in parts):
            return True
        if len(parts) < 4:
            return True
    return False


def _validate_target_url(url: str, resolve_host: bool = True) -> tuple[bool, str | None]:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https"):
        return False, f"unsupported scheme: {parts.scheme or 'none'}"
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False, "missing hostname"
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local") or host.endswith(".internal"):
        return False, f"blocked local host: {host}"
    if _is_obfuscated_ip_host(host):
        return False, f"blocked obfuscated ip host: {host}"
    try:
        ipaddress.ip_address(host)
        if _is_forbidden_ip(host):
            return False, f"blocked private/loopback ip: {host}"
        return True, None
    except ValueError:
        pass

    if resolve_host:
        try:
            infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
            for info in infos:
                addr = info[4][0]
                if _is_forbidden_ip(addr):
                    return False, f"hostname resolves to forbidden ip: {addr}"
        except socket.gaierror as exc:
            return False, f"dns resolve failed: {exc}"
    return True, None


@dataclass
class _StreamedHttpResponse:
    status_code: int
    final_url: str
    headers: httpx.Headers
    encoding: str | None
    content: bytes
    size_capped: bool = False
    redirect_blocked_url: str | None = None
    redirect_blocked_reason: str | None = None
    too_many_redirects: bool = False


async def _read_stream_capped(
    resp: httpx.Response,
    max_bytes: int,
    *,
    allow_pdf_cap: bool = False,
) -> tuple[bytes, bool]:
    """流式增量读取响应体，超过字节上限即中止读取防止大包 OOM。"""
    buf = bytearray()
    cap = max_bytes
    checked_magic = False
    capped = False
    async for chunk in resp.aiter_bytes():
        if not chunk:
            continue
        buf.extend(chunk)
        if allow_pdf_cap and not checked_magic and len(buf) >= 5:
            checked_magic = True
            if bytes(buf[:5]) == b"%PDF-":
                cap = max(cap, _MAX_PDF_BYTES)
        if len(buf) >= cap:
            capped = len(buf) > cap
            del buf[cap:]
            break
    return bytes(buf), capped


async def _fetch_stream_with_redirects(
    cli: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
    max_bytes: int = _MAX_BYTES,
    resolve_host: bool = True,
    max_redirects: int = _MAX_REDIRECTS,
    allow_pdf_cap: bool = False,
) -> _StreamedHttpResponse:
    """执行受控逐跳重定向（每跳前置 SSRF 校验）并流式限额读取响应正文。"""
    current_url = url
    for _ in range(max_redirects + 1):
        async with cli.stream(
            "GET",
            current_url,
            headers=headers,
            timeout=timeout_sec,
            follow_redirects=False,
        ) as resp:
            if resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("location", "").strip()
                if loc:
                    next_url = urljoin(current_url, loc)
                    redir_ok, redir_reason = _validate_target_url(next_url, resolve_host=resolve_host)
                    if not redir_ok:
                        return _StreamedHttpResponse(
                            status_code=resp.status_code,
                            final_url=next_url,
                            headers=resp.headers,
                            encoding=resp.encoding,
                            content=b"",
                            redirect_blocked_url=next_url,
                            redirect_blocked_reason=redir_reason,
                        )
                    current_url = next_url
                    continue
            ctype_low = resp.headers.get("content-type", "").lower()
            eff_cap = max(max_bytes, _MAX_PDF_BYTES) if (allow_pdf_cap and "application/pdf" in ctype_low) else max_bytes
            raw_bytes, capped = await _read_stream_capped(resp, eff_cap, allow_pdf_cap=allow_pdf_cap)
            return _StreamedHttpResponse(
                status_code=resp.status_code,
                final_url=str(resp.url) or current_url,
                headers=resp.headers,
                encoding=resp.encoding,
                content=raw_bytes,
                size_capped=capped,
            )

    return _StreamedHttpResponse(
        status_code=0,
        final_url=current_url,
        headers=httpx.Headers(),
        encoding=None,
        content=b"",
        too_many_redirects=True,
    )
