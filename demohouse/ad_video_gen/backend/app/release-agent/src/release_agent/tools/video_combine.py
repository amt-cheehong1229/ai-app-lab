from __future__ import annotations

import contextlib
import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from moviepy import CompositeVideoClip, VideoFileClip

from ad_video_gen_runtime import build_public_file_url, get_local_media_root
from veadk.utils.logger import get_logger

logger = get_logger(__name__)

shorten_url_service_url = os.getenv("SHORTEN_URL_SERVICE_URL")


async def resolve_short_url(short_url: str) -> str:
    if not shorten_url_service_url:
        return short_url
    try:
        parsed_url = urlparse(short_url)
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "t":
            async with aiohttp.ClientSession() as session:
                async with session.get(short_url) as response:
                    if response.status == 200:
                        return (await response.text()).strip().strip('"')
    except Exception as exc:
        logger.warning("Failed to resolve short URL %s: %s", short_url, exc)
    return short_url


async def _download_video(session: aiohttp.ClientSession, url: str, target_dir: Path) -> Path:
    async with session.get(url, allow_redirects=True) as response:
        response.raise_for_status()
        suffix = Path(urlparse(url).path).suffix or ".mp4"
        path = target_dir / f"{uuid.uuid4().hex}{suffix}"
        path.write_bytes(await response.read())
        return path


async def video_combine(video_urls: list[str]) -> str | None:
    video_urls = [url for url in (video_urls or []) if isinstance(url, str) and url]
    if not video_urls:
        logger.error("No videos found to merge")
        return None

    output_dir = get_local_media_root() / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(dir=output_dir))

    downloaded_files: list[Path] = []
    for index, video_url in enumerate(video_urls):
        video_urls[index] = await resolve_short_url(video_url)

    async with aiohttp.ClientSession() as session:
        for video_url in video_urls:
            parsed = urlparse(video_url)
            if parsed.scheme not in {"http", "https"}:
                logger.warning("Skip non-http(s) video URL: %s", video_url)
                continue
            try:
                downloaded_files.append(await _download_video(session, video_url, temp_dir))
            except Exception as exc:
                logger.error("Failed to download %s: %s", video_url, exc)
                return None

    if not downloaded_files:
        logger.error("No videos were successfully downloaded")
        return None

    clips = []
    final_clip = None
    try:
        start_time = 0.0
        composite_parts = []
        for file_path in downloaded_files:
            clip = VideoFileClip(str(file_path))
            clips.append(clip)
            composite_parts.append(clip.with_start(start_time).with_position("center"))
            start_time += clip.duration

        final_clip = CompositeVideoClip(composite_parts)
        output_path = output_dir / f"merged_video_{uuid.uuid4().hex}.mp4"
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            threads=4,
            logger=None,
        )
        return build_public_file_url(output_path)
    except Exception as exc:
        logger.error("Failed to merge videos: %s", exc)
        return None
    finally:
        for clip in clips:
            with contextlib.suppress(Exception):
                clip.close()
        if final_clip is not None:
            with contextlib.suppress(Exception):
                final_clip.close()
