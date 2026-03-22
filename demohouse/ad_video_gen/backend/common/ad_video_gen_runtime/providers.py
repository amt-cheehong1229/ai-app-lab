from __future__ import annotations

import copy
import os
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
DEFAULT_TEXT_MODEL = "gpt-4.1-mini"
DEFAULT_IMAGE_MODEL = "gpt-image-1"
DEFAULT_VIDEO_MODEL = "sora-2"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"


def _clean_base_url(value: str | None, default: str) -> str:
    return (value or default).rstrip("/")


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_openai_api_key() -> str:
    return _require_env("OPENAI_API_KEY")


def get_openai_base_url() -> str:
    return _clean_base_url(os.getenv("OPENAI_BASE_URL"), DEFAULT_OPENAI_BASE_URL)


def get_perplexity_api_key() -> str:
    return _require_env("PERPLEXITY_API_KEY")


def get_perplexity_base_url() -> str:
    return _clean_base_url(
        os.getenv("PERPLEXITY_BASE_URL"), DEFAULT_PERPLEXITY_BASE_URL
    )


def get_text_model() -> str:
    return os.getenv("OPENAI_MODEL_TEXT", DEFAULT_TEXT_MODEL)


def get_vision_model() -> str:
    return os.getenv("OPENAI_MODEL_VISION", get_text_model())


def get_eval_model() -> str:
    return os.getenv("OPENAI_MODEL_EVAL", get_vision_model())


def get_format_model() -> str:
    return os.getenv("OPENAI_MODEL_FORMAT", get_eval_model())


def get_image_model() -> str:
    return os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)


def get_video_model() -> str:
    return os.getenv("OPENAI_VIDEO_MODEL", DEFAULT_VIDEO_MODEL)


def get_perplexity_model() -> str:
    return os.getenv("PERPLEXITY_MODEL", DEFAULT_PERPLEXITY_MODEL)


@lru_cache(maxsize=1)
def build_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=get_openai_api_key(),
        base_url=get_openai_base_url(),
    )


def maybe_model_extra_config(*_: Any, **__: Any) -> dict[str, Any]:
    # The original demo relied on Ark-only fields like extra_body.thinking.
    # The OpenAI-first local build keeps this empty to stay provider-neutral.
    return {}


def build_strict_json_schema(model: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(model, dict):
        schema = copy.deepcopy(model)
    else:
        schema = copy.deepcopy(model.model_json_schema())

    def _visit(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return {"$ref": node["$ref"]}

            for unsupported_key in ("default", "examples"):
                node.pop(unsupported_key, None)

            if node.get("type") == "object":
                node["additionalProperties"] = False
                properties = node.get("properties") or {}
                node["properties"] = {key: _visit(value) for key, value in properties.items()}
                node.setdefault("required", list(properties.keys()))

            if node.get("type") == "array" and "items" in node:
                node["items"] = _visit(node["items"])

            for key in ("$defs", "definitions", "patternProperties", "dependentSchemas"):
                if key in node and isinstance(node[key], dict):
                    node[key] = {child_key: _visit(child_value) for child_key, child_value in node[key].items()}

            for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
                if key in node and isinstance(node[key], list):
                    node[key] = [_visit(item) for item in node[key]]

        elif isinstance(node, list):
            return [_visit(item) for item in node]

        return node

    return _visit(schema)
