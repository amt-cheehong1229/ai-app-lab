from __future__ import annotations


async def video_combine(*_, **__) -> dict:
    return {
        "film_url": "",
        "success": False,
        "message": "VOD combine is disabled in the local OpenAI-first build. Use release_agent.tools.video_combine instead.",
    }
