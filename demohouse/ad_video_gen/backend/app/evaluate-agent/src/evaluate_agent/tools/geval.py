from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import urlparse

import aiohttp

from ad_video_gen_runtime import (
    build_async_openai_client,
    build_strict_json_schema,
    convert_image_url_to_data_url,
    extract_video_frames_as_data_urls,
    get_eval_model,
)
from evaluate_agent.prompt import PROMPT_EVALUATE_ITEM_AGENT
from evaluate_agent.utils.types import EvaluationList, ScoredImageList, ScoredVideoList
from veadk.utils.logger import get_logger

evaluate_agent_instruction = PROMPT_EVALUATE_ITEM_AGENT
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


async def repair_evaluate_input(
    media_list: list[dict[str, Any]], media_type: str = "image"
) -> list[list[dict[str, Any]]]:
    result = []
    for shot in media_list:
        shot_id = shot.get("shot_id", "")
        reference_media_list = shot.get("reference", [])
        if isinstance(reference_media_list, str):
            reference_media_list = [reference_media_list]
        media_url_list = [item["url"] for item in shot.get("media", [])]

        resolved_references = []
        for reference_media in reference_media_list:
            if str(reference_media).strip():
                resolved_reference = await resolve_short_url(reference_media)
                resolved_references.append(
                    await convert_image_url_to_data_url(resolved_reference)
                )

        for media_id, media_url in enumerate(media_url_list):
            resolved_media_url = await resolve_short_url(media_url)
            content = []
            if media_type == "image":
                resolved_media_url = await convert_image_url_to_data_url(
                    resolved_media_url
                )
                content.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"你正在评估一张分镜图片。shot_id={shot_id}，media_id={media_id}。"
                            "第1张图是待评估图片，后续图片是参考图。请严格输出打分结果。"
                        ),
                    }
                )
                content.append(
                    {"type": "input_image", "image_url": resolved_media_url}
                )
            else:
                content.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"你正在评估一段分镜视频。shot_id={shot_id}，media_id={media_id}。"
                            "接下来你会收到这段视频抽取出的若干关键帧，再附上参考图片。"
                            "请综合这些关键帧来判断视频质量。"
                        ),
                    }
                )
                frame_data_urls = await extract_video_frames_as_data_urls(
                    resolved_media_url, max_frames=3
                )
                for data_url in frame_data_urls:
                    content.append({"type": "input_image", "image_url": data_url})

            for reference_url in resolved_references:
                content.append({"type": "input_image", "image_url": reference_url})
            result.append({"role": "user", "content": content})

    return result


async def evaluate_media(
    media_list: list[dict[str, Any]],
    media_type: str = "image",
) -> dict:
    """
    Evaluate a whole storyboard media list in one tool call.

    Preferred usage:
    - image task: evaluate_media(media_list=image_list, media_type="image")
    - video task: evaluate_media(media_list=video_list, media_type="video")
    """
    normalized_media_list: list[dict[str, Any]] = []
    source_key = "images" if media_type == "image" else "videos"
    for shot in media_list:
        if not isinstance(shot, dict):
            continue
        if "media" in shot:
            media_entries = shot.get("media") or []
        else:
            media_entries = shot.get(source_key) or []
        normalized_media_list.append(
            {
                "shot_id": shot.get("shot_id", ""),
                "prompt": shot.get("prompt", ""),
                "action": shot.get("action", ""),
                "reference": shot.get("reference", ""),
                "words": shot.get("words", ""),
                "media": media_entries,
            }
        )

    logger.debug(
        "Start to evaluate {} list: items={}",
        media_type,
        len(normalized_media_list),
    )
    message_content = await repair_evaluate_input(
        normalized_media_list, media_type=media_type
    )
    client = build_async_openai_client()

    async def process_message(message: dict[str, Any]) -> dict[str, Any]:
        response = await client.responses.create(
            model=get_eval_model(),
            instructions=evaluate_agent_instruction,
            input=[message],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "EvaluationList",
                    "schema": build_strict_json_schema(EvaluationList),
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text).get("evaluation", {})

    result = await asyncio.gather(*(process_message(msg) for msg in message_content))

    merged_result = {}
    for item in result:
        shot_id = item.get("shot_id")
        media_id = int(item.get("media_id", 0))
        merged_result.setdefault(shot_id, {"shot_id": shot_id, "items": []})
        merged_result[shot_id]["items"].append(
            (media_id, item.get("scores"), item.get("reason"))
        )

    shot_index = {shot.get("shot_id", ""): shot for shot in normalized_media_list}

    def normalize_reference(ref_val):
        if isinstance(ref_val, list):
            return ",".join(ref_val)
        return ref_val or ""

    if media_type == "image":
        scored_image_list = []
        for shot_id, data in merged_result.items():
            shot = shot_index.get(shot_id, {})
            media_entries = shot.get("media", [])
            eval_map = {mi: (score, reason) for mi, score, reason in data["items"]}
            image_items = []
            for idx, media in enumerate(media_entries):
                if idx not in eval_map:
                    continue
                score, reason = eval_map[idx]
                image_items.append(
                    {
                        "id": int(media.get("id", idx)),
                        "url": media.get("url", ""),
                        "score": float(score) if score is not None else 0.0,
                        "reason": reason or "",
                    }
                )
            scored_image_list.append(
                {
                    "shot_id": shot_id,
                    "prompt": shot.get("prompt", ""),
                    "action": shot.get("action", ""),
                    "reference": normalize_reference(shot.get("reference")),
                    "words": shot.get("words", ""),
                    "images": image_items,
                }
            )
        output = {
            "scored_image_list": scored_image_list,
            "status": {"success": True, "message": ""},
        }
        try:
            return ScoredImageList.model_validate(output).model_dump()
        except Exception:
            return output

    scored_video_list = []
    for shot_id, data in merged_result.items():
        shot = shot_index.get(shot_id, {})
        media_entries = shot.get("media", [])
        eval_map = {mi: (score, reason) for mi, score, reason in data["items"]}
        video_items = []
        for idx, media in enumerate(media_entries):
            if idx not in eval_map:
                continue
            score, reason = eval_map[idx]
            video_items.append(
                {
                    "id": int(media.get("id", idx)),
                    "url": media.get("url", ""),
                    "score": float(score) if score is not None else 0.0,
                    "reason": reason or "",
                }
            )
        scored_video_list.append(
            {
                "shot_id": shot_id,
                "prompt": shot.get("prompt", ""),
                "action": shot.get("action", ""),
                "reference": normalize_reference(shot.get("reference")),
                "words": shot.get("words", ""),
                "videos": video_items,
            }
        )

    output = {
        "scored_video_list": scored_video_list,
        "status": {"success": True, "message": ""},
    }
    try:
        return ScoredVideoList.model_validate(output).model_dump()
    except Exception:
        return output
