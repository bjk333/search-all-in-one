"""media_meta: 纯 Python 零依赖本地多模态媒体头元数据解析（图像/EXIF/WAV/MP4）。"""

from __future__ import annotations

import pathlib
import re
import struct
from typing import Any

_MAX_HEADER_BYTES = 128 * 1024
_EXIF_DATE_RE = re.compile(rb"(?:19|20)\d{2}:[01]\d:[0-3]\d")
_MP4_EXTS = {".mp4", ".mov", ".m4a", ".m4v", ".3gp"}
_JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF,
}


def _enrich_dimensions(meta: dict[str, Any], width: int, height: int) -> None:
    if not (1 <= width <= 100000 and 1 <= height <= 100000):
        return
    meta["width"] = width
    meta["height"] = height
    if width > height:
        meta["orientation"] = "landscape"
    elif height > width:
        meta["orientation"] = "portrait"
    else:
        meta["orientation"] = "square"

    long_side = max(width, height)
    short_side = min(width, height)
    if long_side >= 3840 or short_side >= 2160:
        meta["resolution"] = "4k"
    elif long_side >= 1920 or short_side >= 1080:
        meta["resolution"] = "1080p"
    elif long_side >= 1280 or short_side >= 720:
        meta["resolution"] = "720p"


def _parse_png(data: bytes) -> dict[str, Any]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return {}
    w, h = struct.unpack(">II", data[16:24])
    meta: dict[str, Any] = {}
    _enrich_dimensions(meta, w, h)
    return meta


def _parse_gif(data: bytes) -> dict[str, Any]:
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return {}
    w, h = struct.unpack("<HH", data[6:10])
    meta: dict[str, Any] = {}
    _enrich_dimensions(meta, w, h)
    return meta


def _parse_bmp(data: bytes) -> dict[str, Any]:
    if len(data) < 26 or data[:2] != b"BM":
        return {}
    dib_size = struct.unpack("<I", data[14:18])[0]
    if dib_size >= 40 and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        h = abs(h)
    elif dib_size == 12 and len(data) >= 22:
        w, h = struct.unpack("<HH", data[18:22])
    else:
        return {}
    meta: dict[str, Any] = {}
    _enrich_dimensions(meta, w, h)
    return meta


def _parse_jpeg(data: bytes) -> dict[str, Any]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return {}
    meta: dict[str, Any] = {}
    pos = 2
    n = len(data)
    while pos < n - 1:
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = data[pos]
        pos += 1
        if marker in (0xD9, 0xDA):  # EOI / SOS
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if pos + 2 > n:
            break
        seg_len = struct.unpack(">H", data[pos : pos + 2])[0]
        if seg_len < 2 or pos + seg_len > n:
            break
        seg = data[pos + 2 : pos + seg_len]
        if marker in _JPEG_SOF_MARKERS and len(seg) >= 5 and "width" not in meta:
            h, w = struct.unpack(">HH", seg[1:5])
            _enrich_dimensions(meta, w, h)
        elif marker == 0xE1 and "captured_date" not in meta:
            m = _EXIF_DATE_RE.search(seg)
            if m:
                raw_date = m.group(0).decode("ascii", errors="ignore")
                yr, mo, dy = raw_date.split(":")
                if 1 <= int(mo) <= 12 and 1 <= int(dy) <= 31:
                    meta["captured_date"] = f"{yr}-{mo}-{dy}"
        pos += seg_len
    return meta


def _parse_webp(data: bytes) -> dict[str, Any]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return {}
    chunk = data[12:16]
    w = h = 0
    if chunk == b"VP8X" and len(data) >= 30:
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
    elif chunk == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
    elif chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = struct.unpack("<I", data[21:25])[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
    else:
        return {}
    meta: dict[str, Any] = {}
    _enrich_dimensions(meta, w, h)
    return meta


def _parse_wav(data: bytes) -> dict[str, Any]:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {}
    pos = 12
    n = len(data)
    sample_rate = 0
    channels = 0
    byte_rate = 0
    data_size = 0
    while pos + 8 <= n:
        cid = data[pos : pos + 4]
        csize = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        payload_start = pos + 8
        if cid == b"fmt " and csize >= 16 and payload_start + 16 <= n:
            _, channels, sample_rate, byte_rate = struct.unpack("<HHII", data[payload_start : payload_start + 12])
        elif cid == b"data":
            data_size = csize
            if sample_rate > 0 and byte_rate > 0:
                break
        step = csize + (csize & 1)
        if step <= 0 and csize == 0:
            pos = payload_start
        else:
            pos = payload_start + step
    meta: dict[str, Any] = {}
    if sample_rate > 0:
        meta["sample_rate"] = sample_rate
    if channels > 0:
        meta["channels"] = channels
    if byte_rate > 0 and data_size > 0:
        meta["duration_sec"] = round(data_size / byte_rate, 2)
    return meta


def _parse_mvhd_duration(moov_bytes: bytes) -> float | None:
    pos = 0
    n = len(moov_bytes)
    while pos + 8 <= n:
        size, atom = struct.unpack(">I4s", moov_bytes[pos : pos + 8])
        hdr = 8
        if size == 1:
            if pos + 16 > n:
                break
            size = struct.unpack(">Q", moov_bytes[pos + 8 : pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = n - pos
        if size < hdr or pos + size > n:
            break
        if atom == b"mvhd":
            payload = moov_bytes[pos + hdr : pos + size]
            if not payload:
                return None
            version = payload[0]
            if version == 0 and len(payload) >= 20:
                timescale, duration = struct.unpack(">II", payload[12:20])
            elif version == 1 and len(payload) >= 32:
                timescale, duration = struct.unpack(">IQ", payload[20:32])
            else:
                return None
            if timescale > 0 and duration > 0:
                return round(duration / timescale, 2)
            return None
        pos += size
    return None


def _parse_mp4_file(path: pathlib.Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    if file_size < 16:
        return {}
    with path.open("rb") as fh:
        pos = 0
        for _ in range(64):
            if pos + 8 > file_size:
                break
            fh.seek(pos)
            header = fh.read(8)
            if len(header) < 8:
                break
            size, atom = struct.unpack(">I4s", header)
            hdr = 8
            if size == 1:
                ext = fh.read(8)
                if len(ext) < 8:
                    break
                size = struct.unpack(">Q", ext)[0]
                hdr = 16
            elif size == 0:
                size = file_size - pos
            if size < hdr or pos + size > file_size:
                break
            if atom == b"moov":
                body_len = min(size - hdr, 512 * 1024)
                moov_body = fh.read(body_len)
                dur = _parse_mvhd_duration(moov_body)
                return {"duration_sec": dur} if dur is not None else {}
            pos += size
    return {}


def extract_media_metadata(path: pathlib.Path) -> dict[str, Any]:
    """安全解析常见本地图像/音视频头部元数据；任意损坏或不支持格式返回 {}。"""
    try:
        with path.open("rb") as fh:
            head = fh.read(_MAX_HEADER_BYTES)
    except OSError:
        return {}
    if len(head) < 8:
        return {}

    try:
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return _parse_png(head)
        if head[:6] in (b"GIF87a", b"GIF89a"):
            return _parse_gif(head)
        if head[:2] == b"BM":
            return _parse_bmp(head)
        if head[:2] == b"\xff\xd8":
            return _parse_jpeg(head)
        if head[:4] == b"RIFF" and len(head) >= 12:
            if head[8:12] == b"WEBP":
                return _parse_webp(head)
            if head[8:12] == b"WAVE":
                return _parse_wav(head)
        if path.suffix.lower() in _MP4_EXTS or (len(head) >= 12 and head[4:8] in (b"ftyp", b"moov")):
            return _parse_mp4_file(path)
    except Exception:
        return {}
    return {}


def format_media_metadata_text(meta: dict[str, Any]) -> str:
    """将结构化媒体头元数据渲染为可检索、可读的高信噪比短语。"""
    if not meta:
        return ""
    parts: list[str] = []
    if "width" in meta and "height" in meta:
        parts.append(f"{meta['width']}x{meta['height']}")
    if meta.get("orientation"):
        parts.append(str(meta["orientation"]))
    if meta.get("resolution"):
        parts.append(str(meta["resolution"]))
    if meta.get("captured_date"):
        parts.append(f"captured:{meta['captured_date']}")
    if "duration_sec" in meta and meta["duration_sec"] is not None:
        parts.append(f"{float(meta['duration_sec']):.1f}s")
    if meta.get("sample_rate"):
        parts.append(f"{meta['sample_rate']}Hz")
    if meta.get("channels"):
        parts.append(f"{meta['channels']}ch")
    return " ".join(parts)
