"""LLM-powered brain science assistant.

Supports multiple providers:
- opencodego (default): OpenAI-compatible, model mimo-v2.5 via opencode.ai
- dashscope: Alibaba Qwen via dashscope API (legacy)
"""

import os
from http import HTTPStatus
from typing import Literal

from pybnt.core.logconf import logger

Provider = Literal["opencodego", "dashscope"]

OPENAICODEGO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_PROVIDER: Provider = "opencodego"
DEFAULT_MODEL = "mimo-v2.5"
DASHSCOPE_DEFAULT_MODEL = "qwen-plus"


def _get_opencode_client():
    """Get OpenCodeGo OpenAI-compatible client."""
    from openai import OpenAI

    api_key = os.environ.get("OPENCODEGO_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENCODEGO_API_KEY not set. Get your key from https://opencode.ai"
        )
    return OpenAI(base_url=OPENAICODEGO_BASE_URL, api_key=api_key)


def ask_llm_openode(
    message: list[dict],
    model: str = DEFAULT_MODEL,
    stream: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str | None:
    """Ask LLM via OpenCodeGo provider.

    Args:
        message: List of dicts with role/content
        model: Model name (default: mimo-v2.5)
        stream: Enable streaming response
        max_tokens: Max tokens in response
        temperature: Sampling temperature

    Returns:
        Response text or None on error
    """
    try:
        client = _get_opencode_client()
        kwargs = {
            "model": model,
            "messages": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stream:
            response = client.chat.completions.create(**kwargs, stream=True)
            result = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
            return result
        else:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return None
    except Exception as e:
        logger.error("OpenCodeGo API error: %s", e)
        return None


def ask_llm_dashscope(
    message: list[dict],
    model: str = DASHSCOPE_DEFAULT_MODEL,
    stream: bool = False,
) -> str | None:
    """Ask LLM via DashScope (Alibaba Qwen) provider."""
    import dashscope

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY not set")
        return None

    dashscope.api_key = api_key
    kwargs = {
        "model": model,
        "messages": message,
        "result_format": "message",
    }
    if stream:
        kwargs["stream"] = True
        response = dashscope.Generation.call(**kwargs)
        result = ""
        for chunk in response:
            if hasattr(chunk, "output") and chunk.output and hasattr(chunk.output, "text"):
                part = chunk.output.text
                result += part
                print(part, end="", flush=True)
        print()
        return result
    response = dashscope.Generation.call(**kwargs)
    if response.status_code == HTTPStatus.OK:
        return response.output.text if hasattr(response, "output") else str(response)
    else:
        logger.error(
            "DashScope error: %s - %s", response.status_code, response.message
        )
        return None


def ask_llm(
    message: list[dict],
    api_key: str = "",
    model: str | None = None,
    stream: bool = False,
    provider: Provider = DEFAULT_PROVIDER,
    max_tokens: int = 4096,
) -> str | None:
    """Ask LLM about brain science questions.

    Args:
        message: List of dicts with role/content
        api_key: API key (for backward compat, sets appropriate env var)
        model: Model name (default depends on provider)
        stream: Enable streaming response
        provider: Provider to use - "opencodego" (default) or "dashscope"
        max_tokens: Max tokens in response

    Returns:
        Response text or None on error
    """
    # Handle backward-compat api_key parameter
    if api_key:
        if provider == "opencodego":
            os.environ["OPENCODEGO_API_KEY"] = api_key
        else:
            os.environ["DASHSCOPE_API_KEY"] = api_key

    if model is None:
        model = DEFAULT_MODEL if provider == "opencodego" else DASHSCOPE_DEFAULT_MODEL

    if provider == "opencodego":
        return ask_llm_openode(message, model, stream, max_tokens)
    elif provider == "dashscope":
        return ask_llm_dashscope(message, model, stream)
    else:
        logger.error("Unknown provider: %s", provider)
        return None


def ask_llm_multimodal(
    image_path: str,
    prompt: str = "Describe this brain image",
    provider: Provider = DEFAULT_PROVIDER,
    model: str | None = None,
) -> str | None:
    """Send an image for multimodal analysis.

    Works with MiMo-V2.5 (supports images) via OpenCodeGo.
    """
    import base64

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()[1:] or "png"
    mime_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "nii": "application/octet-stream",
    }.get(ext, "image/png")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
            ],
        }
    ]

    if provider == "opencodego":
        if model is None:
            model = "mimo-v2.5"  # multimodal
        return ask_llm_openode(messages, model, max_tokens=2048)
    else:
        logger.warning(
            "DashScope does not support multimodal. Use opencodego provider."
        )
        return None


# Backward compatibility alias
call = ask_llm
