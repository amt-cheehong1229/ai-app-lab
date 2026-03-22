from __future__ import annotations

import os


def patch_veadk_for_openai() -> None:
    """
    VeADK injects Ark-oriented headers/body fields by default.
    For the local OpenAI-first build we clear those defaults before agents are
    instantiated, while keeping the rest of VeADK intact.
    """
    provider = os.getenv("MODEL_AGENT_PROVIDER", "openai").strip().lower()
    if provider != "openai":
        return

    import veadk.consts as veadk_consts

    veadk_consts.DEFAULT_MODEL_EXTRA_CONFIG = {
        "extra_headers": {},
        "extra_body": {},
    }
