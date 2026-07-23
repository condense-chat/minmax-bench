"""AWS Bedrock as an upstream transport.

Bedrock's Anthropic-compatible endpoint (``/anthropic/v1/messages`` on
``bedrock-runtime``) speaks the exact messages dialect — including
``cache_control`` and cache-aware usage — so it can stand in for
api.anthropic.com wherever the bench itself makes the model call. Only two
things differ: auth is a short-lived bearer token derived from the AWS
credential chain, and model ids must be inference-profile ids
(``global.anthropic.claude-sonnet-5``); plain ids are rejected.
"""

from __future__ import annotations

import re
from functools import lru_cache

_DATE_SUFFIX = re.compile(r"-(20\d{6}|latest)$")


def anthropic_base(region: str) -> str:
    return f"https://bedrock-runtime.{region}.amazonaws.com/anthropic"


def bedrock_model_id(model: str) -> str:
    """Map an Anthropic model id to a Bedrock global inference-profile id."""
    if model.startswith(("global.", "eu.", "us.", "anthropic.")):
        return model
    return "global.anthropic." + _DATE_SUFFIX.sub("", model)


@lru_cache(maxsize=4)
def bearer_token(region: str) -> str:
    """Token from the default AWS credential chain; valid 12h, so one per run."""
    from aws_bedrock_token_generator import provide_token

    return provide_token(region=region)
