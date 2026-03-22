from __future__ import annotations

import asyncio
import contextlib
import os
import re
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import aiohttp

from ad_video_gen_runtime import (
    create_placeholder_image,
    create_still_video_from_image,
    download_to_temp_file,
    get_openai_api_key,
    get_openai_base_url,
    get_video_model,
    save_binary_file,
)
from google.adk.tools import ToolContext
from veadk.utils.logger import get_logger

logger = get_logger(__name__)

shorten_url_service_url = os.getenv("SHORTEN_URL_SERVICE_URL")
VIDEO_STATUS_RETRYABLE = {"queued", "in_progress"}
VIDEO_SIZE_MAP = {
    "16:9": "1280x720",
    "21:9": "1792x1024",
    "4:3": "1792x1024",
    "3:2": "1792x1024",
    "9:16": "720x1280",
    "3:4": "1024x1792",
    "2:3": "1024x1792",
    "1:1": "1024x1792",
}


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


def _normalize_seconds() -> str:
    raw_value = int(os.getenv("OPENAI_VIDEO_SECONDS", "4"))
    if raw_value <= 4:
        return "4"
    if raw_value <= 8:
        return "8"
    return "12"


def _pick_video_size(prompt: str) -> str:
    match = re.search(r"--(?:rt|ratio)\s+([0-9]+:[0-9]+)", prompt)
    if not match:
        return "1280x720"
    ratio = match.group(1)
    return VIDEO_SIZE_MAP.get(ratio, "1280x720")


def _strip_prompt_flags(prompt: str) -> str:
    prompt = re.sub(r"--(?:rs|resolution)\s+\S+", "", prompt)
    prompt = re.sub(r"--(?:rt|ratio)\s+\S+", "", prompt)
    prompt = re.sub(r"--(?:fps|framespersecond)\s+\S+", "", prompt)
    prompt = re.sub(r"--(?:wm|watermark)\s+\S+", "", prompt)
    prompt = re.sub(r"--seed\s+\S+", "", prompt)
    prompt = re.sub(r"--(?:cf|camerafixed)\s+\S+", "", prompt)
    return " ".join(prompt.split())


async def _create_openai_video(prompt: str, first_frame: str | None = None) -> str:
    api_key = get_openai_api_key()
    base_url = get_openai_base_url()
    video_size = _pick_video_size(prompt)
    cleaned_prompt = _strip_prompt_flags(prompt)

    temp_reference_path: Path | None = None
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field("model", get_video_model())
        form.add_field("prompt", cleaned_prompt)
        form.add_field("seconds", _normalize_seconds())
        form.add_field("size", video_size)

        if first_frame:
            temp_reference_path = await download_to_temp_file(first_frame, suffix=".png")
            form.add_field(
                "input_reference",
                temp_reference_path.read_bytes(),
                filename=temp_reference_path.name,
                content_type="image/png",
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with session.post(
                f"{base_url}/videos",
                data=form,
                headers=headers,
            ) as response:
                response.raise_for_status()
                video_job = await response.json()
        finally:
            if temp_reference_path is not None:
                with contextlib.suppress(Exception):
                    temp_reference_path.unlink(missing_ok=True)

        poll_interval = float(os.getenv("OPENAI_VIDEO_POLL_INTERVAL", "5"))
        max_retries = int(os.getenv("OPENAI_VIDEO_MAX_RETRIES", "60"))
        for _ in range(max_retries):
            async with session.get(
                f"{base_url}/videos/{video_job['id']}",
                headers=headers,
            ) as response:
                response.raise_for_status()
                status_payload = await response.json()
            status = status_payload.get("status")
            if status == "completed":
                async with session.get(
                    f"{base_url}/videos/{video_job['id']}/content",
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    binary = await response.read()
                _, public_url = save_binary_file(
                    binary,
                    subdir="videos",
                    suffix=".mp4",
                )
                return public_url
            if status == "failed":
                raise RuntimeError(status_payload.get("error", "Video generation failed"))
            if status not in VIDEO_STATUS_RETRYABLE:
                raise RuntimeError(f"Unexpected video status: {status}")
            await asyncio.sleep(poll_interval)

    raise TimeoutError("Timed out while waiting for OpenAI video generation")


async def _fallback_video(prompt: str, first_frame: str | None = None) -> str:
    if not first_frame:
        first_frame = await create_placeholder_image(prompt, size="1536x1024")
    return await create_still_video_from_image(first_frame, duration_seconds=5)


async def video_generate(
    tasks: list[dict], tool_context: ToolContext, batch_size: int = 32
) -> Dict:
    """
    Generate storyboard videos in one batch.

    Args:
        tasks:
            A required list of video tasks. Each task must be a dict with:
            - video_name (str): unique task name such as `task_0`
            - prompt (str): full video prompt, including motion details
            - first_frame (str, optional): image URL used as the first frame
        batch_size:
            Maximum number of tasks to execute from `params`.

    Important:
    - Always pass `tasks`.
    - Do not call this tool with only `batch_size`.
    """
    success_list: list[dict] = []
    error_list: list[str] = []

    for item in tasks[:batch_size]:
        video_name = item["video_name"]
        prompt = item["prompt"]
        first_frame = item.get("first_frame")

        if first_frame:
            first_frame = await resolve_short_url(first_frame)

        try:
            video_url = await _create_openai_video(prompt, first_frame=first_frame)
        except Exception as exc:
            logger.warning(
                "OpenAI video generation failed for %s, using local fallback: %s",
                video_name,
                exc,
            )
            try:
                video_url = await _fallback_video(prompt, first_frame=first_frame)
            except Exception as fallback_exc:
                logger.error(
                    "Local video fallback failed for %s: %s",
                    video_name,
                    fallback_exc,
                )
                error_list.append(video_name)
                continue

        tool_context.state[f"{video_name}_video_url"] = video_url
        success_list.append({video_name: video_url})

    return {"status": "success" if success_list else "error", "success_list": success_list, "error_list": error_list}
