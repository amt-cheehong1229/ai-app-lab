from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import dotenv_values


BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_ROOT / "app"
ENV_PATH = BACKEND_ROOT / ".env"


def load_env() -> dict[str, str]:
    env = dict(dotenv_values(ENV_PATH))
    for key, value in os.environ.items():
        if value is not None:
            env[key] = value
    return {key: str(value) for key, value in env.items() if value is not None}


def require_any(env: dict[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    raise ValueError(f"Missing required setting. Tried: {', '.join(names)}")


def dump_yaml(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def build_configs(env: dict[str, str]) -> dict[Path, dict]:
    text_model = require_any(env, "OPENAI_MODEL_TEXT")
    vision_model = require_any(env, "OPENAI_MODEL_VISION", "OPENAI_MODEL_TEXT")
    eval_model = require_any(env, "OPENAI_MODEL_EVAL", "OPENAI_MODEL_VISION")
    format_model = env.get("OPENAI_MODEL_FORMAT", eval_model)
    image_model = require_any(env, "OPENAI_IMAGE_MODEL")
    video_model = require_any(env, "OPENAI_VIDEO_MODEL")
    openai_api_key = require_any(env, "OPENAI_API_KEY")
    openai_base_url = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    market_config = {
        "model": {
            "agent": {
                "provider": "openai",
                "name": text_model,
                "api_base": openai_base_url,
                "api_key": openai_api_key,
            },
            "format": {"name": format_model},
        },
        "logging": {"level": env.get("LOG_LEVEL", "DEBUG")},
    }

    director_config = {
        "model": {
            "agent": {
                "provider": "openai",
                "name": text_model,
                "api_base": openai_base_url,
                "api_key": openai_api_key,
            },
            "video": {
                "name": video_model,
                "api_base": openai_base_url,
                "api_key": openai_api_key,
            },
            "image": {
                "name": image_model,
                "api_base": openai_base_url,
                "api_key": openai_api_key,
            },
            "format": {"name": format_model},
        },
        "shorten_url_service_url": env.get(
            "SHORTEN_URL_SERVICE_URL", "http://127.0.0.1:8005"
        ),
        "logging": {"level": env.get("LOG_LEVEL", "DEBUG")},
    }

    evaluate_config = {
        "model": {
            "agent": {
                "provider": "openai",
                "name": vision_model,
                "api_base": openai_base_url,
                "api_key": openai_api_key,
            },
            "format": {"name": format_model},
        },
        "shorten_url_service_url": env.get(
            "SHORTEN_URL_SERVICE_URL", "http://127.0.0.1:8005"
        ),
        "logging": {"level": env.get("LOG_LEVEL", "DEBUG")},
    }

    release_config = {
        "model": {
            "agent": {
                "provider": "openai",
                "name": text_model,
                "api_base": openai_base_url,
                "api_key": openai_api_key,
            },
            "format": {"name": format_model},
        },
        "shorten_url_service_url": env.get(
            "SHORTEN_URL_SERVICE_URL", "http://127.0.0.1:8005"
        ),
        "tools": {"vod": {"enabled": False}},
        "logging": {"level": env.get("LOG_LEVEL", "DEBUG")},
    }

    multimedia_config = {
        "model": {
            "agent": {
                "provider": "openai",
                "name": text_model,
                "api_base": openai_base_url,
                "api_key": openai_api_key,
            }
        },
        "logging": {"level": env.get("LOG_LEVEL", "DEBUG")},
        "remote_agent": {
            "market_agent": {
                "url": env.get(
                    "REMOTE_AGENT_MARKET_AGENT_URL", "http://127.0.0.1:8000"
                )
            },
            "director_agent": {
                "url": env.get(
                    "REMOTE_AGENT_DIRECTOR_AGENT_URL", "http://127.0.0.1:8001"
                )
            },
            "evaluate_agent": {
                "url": env.get(
                    "REMOTE_AGENT_EVALUATE_AGENT_URL", "http://127.0.0.1:8002"
                )
            },
            "release_agent": {
                "url": env.get(
                    "REMOTE_AGENT_RELEASE_AGENT_URL", "http://127.0.0.1:8003"
                )
            },
        },
    }

    return {
        APP_ROOT / "market-agent" / "config.yaml": market_config,
        APP_ROOT / "director-agent" / "config.yaml": director_config,
        APP_ROOT / "evaluate-agent" / "config.yaml": evaluate_config,
        APP_ROOT / "release-agent" / "config.yaml": release_config,
        APP_ROOT / "multimedia-agent" / "config.yaml": multimedia_config,
    }


def main() -> None:
    env = load_env()
    if not env.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required. Fill backend/.env first.")
    if not env.get("PERPLEXITY_API_KEY"):
        print("Warning: PERPLEXITY_API_KEY is empty, market search will fail at runtime.")

    media_root = env.get("LOCAL_MEDIA_DIR")
    if not media_root:
        media_root = str(BACKEND_ROOT / ".local_media")
        env["LOCAL_MEDIA_DIR"] = media_root

    Path(media_root).mkdir(parents=True, exist_ok=True)

    for path, payload in build_configs(env).items():
        dump_yaml(path, payload)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
