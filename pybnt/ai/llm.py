"""LLM-powered brain science assistant."""

import os
from http import HTTPStatus


def ask_llm(message: list[dict], api_key: str = "", model: str = "qwen-plus",
            stream: bool = False) -> str | None:
    """Ask LLM about brain science questions.
    - message: list of dicts with role/content
    - api_key: API key (reads from env DASHSCOPE_API_KEY if empty)
    - model: model name (default: qwen-plus)
    - stream: enable streaming response
    Returns response text or None on error.
    """
    os.environ["DASHSCOPE_API_KEY"] = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    import dashscope
    dashscope.api_key = api_key
    response = dashscope.Generation.call(
        model=model,
        messages=message,
        result_format="message",
    )
    if response.status_code == HTTPStatus.OK:
        return response
    else:
        print(f"Error: {response.status_code} - {response.message}")
        return None


# Alias for backward compatibility
call = ask_llm
