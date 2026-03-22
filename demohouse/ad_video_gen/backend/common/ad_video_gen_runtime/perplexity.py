from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from .providers import (
    get_perplexity_api_key,
    get_perplexity_base_url,
    get_perplexity_model,
)


SEARCH_SYSTEM_PROMPT = """
你是一個市場研究助手。
請優先回覆和使用者問題直接相關的結論，並保留可驗證的來源連結。
輸出請簡潔、資訊密度高、避免空泛行銷話術。
""".strip()


@lru_cache(maxsize=1)
def _get_perplexity_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=get_perplexity_api_key(),
        base_url=get_perplexity_base_url(),
    )


async def perplexity_search(
    query: str,
    max_results: int = 5,
) -> dict:
    """
    Search the web through Perplexity's OpenAI-compatible chat API.
    """
    client = _get_perplexity_client()
    response = await client.chat.completions.create(
        model=get_perplexity_model(),
        temperature=0.1,
        messages=[
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    message = response.choices[0].message
    citations = list(getattr(message, "citations", []) or [])
    return {
        "query": query,
        "answer": message.content or "",
        "citations": citations[:max_results],
        "status": {"success": True, "message": ""},
    }
