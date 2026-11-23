import struct

import pytest

from omni_search.local_index import LocalIndexEngine, index_paths


def _make_png(w: int, h: int) -> bytes:
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00"


def _make_gif(w: int, h: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", w, h) + b"\x00\x00\x00"


def _make_bmp(w: int, h: int) -> bytes:
    return b"BM" + b"\x00" * 12 + struct.pack("<Iii", 40, w, h) + b"\x00" * 28


def _make_jpeg_with_exif(w: int, h: int, exif_ascii: bytes = b"Exif\x00\x00Canon EOS R5 2026:10:15 12:00:00") -> bytes:
    app1 = b"\xff\xe1" + struct.pack(">H", len(exif_ascii) + 2) + exif_ascii
    sof0_payload = struct.pack(">BHHB", 8, h, w, 3) + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    sof0 = b"\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload
    return b"\xff\xd8" + app1 + sof0 + b"\xff\xd9"


def _make_webp_vp8x(w: int, h: int) -> bytes:
    wm1 = (w - 1).to_bytes(3, "little")
    hm1 = (h - 1).to_bytes(3, "little")
    vp8x_payload = b"\x00\x00\x00\x00" + wm1 + hm1
    body = b"WEBP" + b"VP8X" + struct.pack("<I", len(vp8x_payload)) + vp8x_payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _make_wav(sample_rate: int = 44100, channels: int = 2, bits: int = 16, duration_sec: float = 2.0) -> bytes:
    byte_rate = sample_rate * channels * (bits // 8)
    data_size = int(byte_rate * duration_sec)
    fmt_chunk = b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, channels * (bits // 8), bits)
    data_header = b"data" + struct.pack("<I", data_size) + b"\x00" * 32
    return b"RIFF" + struct.pack("<I", 4 + len(fmt_chunk) + len(data_header)) + b"WAVE" + fmt_chunk + data_header


def _make_mp4_mvhd(timescale: int = 1000, duration: int = 15500) -> bytes:
    # mvhd version 0: 1B version + 3B flags + 4B ctime + 4B mtime + 4B timescale + 4B duration
    mvhd_body = struct.pack(">IIIIII", 0, 0, 0, timescale, duration, 0) + b"\x00" * 76
    mvhd = struct.pack(">I", 8 + len(mvhd_body)) + b"mvhd" + mvhd_body
    moov = struct.pack(">I", 8 + len(mvhd)) + b"moov" + mvhd
    ftyp = struct.pack(">I", 16) + b"ftyp" + b"isom\x00\x00\x00\x00"
    return ftyp + moov


def test_extract_image_headers_and_exif(tmp_path):
    from omni_search.media_meta import extract_media_metadata

    png_p = tmp_path / "screen.png"
    png_p.write_bytes(_make_png(1920, 1080))
    m_png = extract_media_metadata(png_p)
    assert m_png["width"] == 1920 and m_png["height"] == 1080
    assert m_png["orientation"] == "landscape"
    assert m_png["resolution"] == "1080p"

    gif_p = tmp_path / "anim.gif"
    gif_p.write_bytes(_make_gif(640, 480))
    assert extract_media_metadata(gif_p)["width"] == 640

    bmp_p = tmp_path / "mobile.bmp"
    bmp_p.write_bytes(_make_bmp(1080, 1920))
    m_bmp = extract_media_metadata(bmp_p)
    assert m_bmp["orientation"] == "portrait"

    jpg_p = tmp_path / "photo.jpg"
    jpg_p.write_bytes(_make_jpeg_with_exif(3840, 2160))
    m_jpg = extract_media_metadata(jpg_p)
    assert m_jpg["width"] == 3840 and m_jpg["height"] == 2160
    assert m_jpg["resolution"] == "4k"
    assert m_jpg.get("captured_date") == "2026-10-15"

    webp_p = tmp_path / "card.webp"
    webp_p.write_bytes(_make_webp_vp8x(800, 600))
    m_webp = extract_media_metadata(webp_p)
    assert m_webp["width"] == 800 and m_webp["height"] == 600


def test_extract_wav_and_mp4_duration_and_corrupted_safety(tmp_path):
    from omni_search.media_meta import extract_media_metadata

    wav_p = tmp_path / "voice.wav"
    wav_p.write_bytes(_make_wav(44100, 2, 16, 2.0))
    m_wav = extract_media_metadata(wav_p)
    assert m_wav["duration_sec"] == pytest.approx(2.0, abs=0.05)
    assert m_wav["sample_rate"] == 44100
    assert m_wav["channels"] == 2

    mp4_p = tmp_path / "clip.mp4"
    mp4_p.write_bytes(_make_mp4_mvhd(1000, 15500))
    m_mp4 = extract_media_metadata(mp4_p)
    assert m_mp4["duration_sec"] == pytest.approx(15.5, abs=0.05)

    # 损坏二进制不得崩溃
    bad_p = tmp_path / "broken.jpg"
    bad_p.write_bytes(b"not a real jpeg file at all")
    assert extract_media_metadata(bad_p) == {}


@pytest.mark.asyncio
async def test_local_index_searches_media_by_resolution_orientation_and_audio_meta(tmp_path):
    ws = tmp_path / "media_ws"
    ws.mkdir()
    (ws / "desktop_shot.png").write_bytes(_make_png(1920, 1080))
    (ws / "phone_shot.bmp").write_bytes(_make_bmp(1080, 1920))
    (ws / "meeting_clip.wav").write_bytes(_make_wav(44100, 2, 16, 12.5))
    (ws / "demo_video.mp4").write_bytes(_make_mp4_mvhd(1000, 65000))

    db_path = tmp_path / "media.db"
    stats = index_paths([ws], db_path=db_path)
    assert stats.indexed == 4

    eng = LocalIndexEngine(db_path=db_path)

    r_1080 = await eng.search("type:image 1920x1080", limit=5)
    assert [x.title for x in r_1080.results] == ["desktop_shot.png"]
    assert "1920x1080" in r_1080.results[0].snippet

    r_port = await eng.search("type:image portrait", limit=5)
    assert [x.title for x in r_port.results] == ["phone_shot.bmp"]

    r_wav = await eng.search("type:audio 44100hz", limit=5)
    assert [x.title for x in r_wav.results] == ["meeting_clip.wav"]
    assert "12.5s" in r_wav.results[0].snippet

    r_vid = await eng.search("type:video 65.0s", limit=5)
    assert [x.title for x in r_vid.results] == ["demo_video.mp4"]


def test_cli_watch_poll_flag_and_single_round(tmp_path, capsys):
    """omni watch --poll --max-rounds 1 应成功解析参数并输出合法 JSON 索引统计。"""
    import json
    from omni_search.cli import main

    ws = tmp_path / "watch_ws"
    ws.mkdir()
    (ws / "shot.png").write_bytes(_make_png(1280, 720))
    db_path = tmp_path / "watch.db"

    with pytest.raises(SystemExit) as exc:
        main(["watch", str(ws), "--db", str(db_path), "--interval", "0.5", "--max-rounds", "1", "--poll"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload["indexed"] == 1
    assert "mode=poll" in captured.err
