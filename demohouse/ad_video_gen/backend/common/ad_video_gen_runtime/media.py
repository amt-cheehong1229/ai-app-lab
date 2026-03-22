from __future__ import annotations

import base64
import io
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urlparse

import aiohttp
from moviepy import ImageClip, VideoFileClip
from PIL import Image, ImageDraw


def get_backend_root() -> Path:
    value = os.getenv("AD_VIDEO_GEN_BACKEND_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def get_local_media_root() -> Path:
    value = os.getenv("LOCAL_MEDIA_DIR")
    root = Path(value).expanduser() if value else get_backend_root() / ".local_media"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def build_public_file_url(path: str | Path) -> str:
    media_root = get_local_media_root()
    file_path = Path(path).resolve()
    relative_path = file_path.relative_to(media_root)
    base_url = os.getenv("LOCAL_MEDIA_BASE_URL") or os.getenv(
        "SHORT_LINK_DOMAIN", "http://127.0.0.1:8005"
    )
    return f"{base_url.rstrip('/')}/files/{quote(relative_path.as_posix(), safe='/')}"


def save_binary_file(
    data: bytes,
    *,
    subdir: str,
    suffix: str,
    stem: str | None = None,
) -> tuple[Path, str]:
    target_dir = get_local_media_root() / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    file_stem = stem or uuid.uuid4().hex
    file_path = target_dir / f"{file_stem}{suffix}"
    file_path.write_bytes(data)
    return file_path, build_public_file_url(file_path)


async def download_bytes(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, allow_redirects=True) as response:
                response.raise_for_status()
                return await response.read()
    if parsed.scheme == "file":
        return Path(parsed.path).read_bytes()
    if Path(url).exists():
        return Path(url).read_bytes()
    raise ValueError(f"Unsupported asset location: {url}")


async def download_to_temp_file(url: str, suffix: str = "") -> Path:
    data = await download_bytes(url)
    fd, tmp_path = tempfile.mkstemp(suffix=suffix or Path(urlparse(url).path).suffix)
    os.close(fd)
    path = Path(tmp_path)
    path.write_bytes(data)
    return path


async def convert_image_url_to_data_url(url: str) -> str:
    if not url:
        return url

    parsed = urlparse(url)
    if url.startswith("data:"):
        return url

    data = await download_bytes(url)
    mime_type, _ = mimetypes.guess_type(parsed.path or "")
    if not mime_type:
        mime_type = "image/png"
    payload = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def _parse_size(size: str | None) -> tuple[int, int]:
    if not size:
        return 1536, 1024
    try:
        width_str, height_str = size.lower().split("x", 1)
        return max(int(width_str), 256), max(int(height_str), 256)
    except Exception:
        return 1536, 1024


async def create_placeholder_image(
    prompt: str,
    *,
    reference_urls: Iterable[str] | None = None,
    size: str | None = None,
) -> str:
    width, height = _parse_size(size)
    canvas = Image.new("RGB", (width, height), color=(241, 238, 230))

    for reference_url in reference_urls or []:
        try:
            ref_bytes = await download_bytes(reference_url)
            ref_image = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
            ref_image.thumbnail((width, height))
            offset = ((width - ref_image.width) // 2, (height - ref_image.height) // 2)
            canvas.paste(ref_image, offset)
            break
        except Exception:
            continue

    # Keep the fallback lightweight and deterministic without needing extra fonts.
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((24, height - 120, width - 24, height - 24), fill=(255, 255, 255))
    draw.text(
        (48, height - 96),
        "Local fallback render",
        fill=(48, 48, 48),
    )

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    _, public_url = save_binary_file(
        output.getvalue(),
        subdir="images",
        suffix=".png",
        stem=f"fallback_{uuid.uuid4().hex}",
    )
    return public_url


async def create_still_video_from_image(
    image_url: str,
    *,
    duration_seconds: int = 5,
    subdir: str = "videos",
) -> str:
    image_path = await download_to_temp_file(image_url, suffix=".png")
    target_dir = get_local_media_root() / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{uuid.uuid4().hex}.mp4"

    clip = ImageClip(str(image_path)).with_duration(duration_seconds)
    try:
        clip.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio=False,
            logger=None,
        )
    finally:
        clip.close()
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass

    return build_public_file_url(output_path)


async def extract_video_frames_as_data_urls(
    video_url: str,
    *,
    max_frames: int = 3,
) -> list[str]:
    video_path = await download_to_temp_file(video_url, suffix=".mp4")
    frames: list[str] = []
    clip = VideoFileClip(str(video_path))
    try:
        duration = max(clip.duration or 0.0, 0.1)
        sample_points = []
        if max_frames <= 1:
            sample_points = [duration / 2]
        else:
            step = duration / (max_frames + 1)
            sample_points = [step * (index + 1) for index in range(max_frames)]
        for timestamp in sample_points:
            frame = clip.get_frame(min(timestamp, max(duration - 0.01, 0)))
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frames.append(f"data:image/png;base64,{payload}")
    finally:
        clip.close()
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass
    return frames
