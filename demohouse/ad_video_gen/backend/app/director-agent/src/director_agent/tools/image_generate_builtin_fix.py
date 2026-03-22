from __future__ import annotations

import base64
import contextlib
from pathlib import Path
from typing import Dict
import uuid

from ad_video_gen_runtime import (
    build_async_openai_client,
    create_placeholder_image,
    download_to_temp_file,
    get_image_model,
    save_binary_file,
)
from veadk.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_IMAGE_SIZES = {
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "1792x1024",
    "1024x1792",
    "auto",
}


def _normalize_size(size: str | None) -> str:
    if not size:
        return "1536x1024"
    size = size.strip().lower()
    if size in {"1k", "2k", "4k"}:
        return "1536x1024"
    if size in ALLOWED_IMAGE_SIZES:
        return size
    return "1536x1024"


def _normalize_reference_images(image_field) -> list[str]:
    if isinstance(image_field, str) and image_field.strip():
        return [image_field.strip()]
    if isinstance(image_field, list):
        return [str(item).strip() for item in image_field if str(item).strip()]
    return []


async def _open_reference_files(reference_urls: list[str]) -> tuple[list[Path], list]:
    temp_paths: list[Path] = []
    handles: list = []
    for reference_url in reference_urls:
        path = await download_to_temp_file(reference_url, suffix=".png")
        temp_paths.append(path)
        handles.append(path.open("rb"))
    return temp_paths, handles


async def _generate_openai_images(task: dict) -> list[str]:
    prompt = str(task.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Image generation prompt is empty")

    reference_urls = _normalize_reference_images(task.get("image"))
    size = _normalize_size(task.get("size"))
    count = max(int(task.get("max_images", 1) or 1), 1)
    client = build_async_openai_client()
    response = None

    temp_paths: list[Path] = []
    handles: list = []
    try:
        if reference_urls:
            temp_paths, handles = await _open_reference_files(reference_urls)
            response = await client.images.edit(
                model=get_image_model(),
                image=handles if len(handles) > 1 else handles[0],
                prompt=prompt,
                input_fidelity="high",
                size=size if size in {"1024x1024", "1536x1024", "1024x1536", "auto"} else "1536x1024",
                quality="medium",
                output_format="png",
                n=count,
            )
        else:
            response = await client.images.generate(
                model=get_image_model(),
                prompt=prompt,
                size=size,
                quality="medium",
                output_format="png",
                n=count,
            )
    finally:
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.close()
        for path in temp_paths:
            with contextlib.suppress(Exception):
                path.unlink(missing_ok=True)

    urls: list[str] = []
    for index, image_data in enumerate(response.data):
        if getattr(image_data, "b64_json", None):
            _, public_url = save_binary_file(
                base64.b64decode(image_data.b64_json),
                subdir="images",
                suffix=".png",
                stem=f"openai_image_{uuid.uuid4().hex}_{index}",
            )
            urls.append(public_url)
        elif getattr(image_data, "url", None):
            urls.append(image_data.url)
    if not urls:
        raise RuntimeError("OpenAI image API returned no images")
    return urls


async def image_generate(tasks: list[dict], tool_context) -> Dict:
    success_list: list[dict] = []
    error_list: list[str] = []

    for idx, task in enumerate(tasks):
        reference_urls = _normalize_reference_images(task.get("image"))
        try:
            urls = await _generate_openai_images(task)
        except Exception as exc:
            logger.warning(
                "OpenAI image generation failed for task {}, using local fallback: {}",
                idx,
                exc,
            )
            try:
                urls = [
                    await create_placeholder_image(
                        str(task.get("prompt", "")).strip() or f"shot {idx}",
                        reference_urls=reference_urls,
                        size=_normalize_size(task.get("size")),
                    )
                ]
            except Exception as fallback_exc:
                logger.error(
                    "Local fallback image generation failed for task {}: {}",
                    idx,
                    fallback_exc,
                )
                error_list.append(f"task_{idx}")
                continue

        for image_index, image_url in enumerate(urls):
            image_name = f"task_{idx}_image_{image_index}"
            tool_context.state[f"{image_name}_url"] = image_url
            success_list.append({image_name: image_url})

    return {
        "status": "success" if success_list else "error",
        "success_list": success_list,
        "error_list": error_list,
    }
