"""Tests for pybnt.ai.llm provider routing and backward compatibility."""
import os
import sys
import pytest
from unittest import mock

from pybnt.ai.llm import (
    ask_llm,
    ask_llm_openode,
    ask_llm_dashscope,
    ask_llm_multimodal,
    call,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DASHSCOPE_DEFAULT_MODEL,
    OPENAICODEGO_BASE_URL,
)


class TestProviderSelectionDefault:
    """Default provider is opencodego with mimo-v2.5."""

    def test_default_provider_is_opencodego(self):
        assert DEFAULT_PROVIDER == "opencodego"
        assert DEFAULT_MODEL == "mimo-v2.5"

    def test_dashscope_model_constant(self):
        assert DASHSCOPE_DEFAULT_MODEL == "qwen-plus"


class TestOpenCodeGoClientInitialization:
    """OpenCodeGo client initialization with correct base URL and API key."""

    def test_get_opencode_client_builds_correct_client(self):
        from pybnt.ai.llm import _get_opencode_client

        with mock.patch.dict(os.environ, {"OPENCODEGO_API_KEY": "test-key-123"}):
            # OpenAI is a lazy import inside _get_opencode_client;
            # patch the source package where it is imported from.
            with mock.patch("openai.OpenAI") as mock_openai:
                client = _get_opencode_client()
                mock_openai.assert_called_once_with(
                    base_url=OPENAICODEGO_BASE_URL,
                    api_key="test-key-123",
                )
                assert client == mock_openai.return_value

    def test_get_opencode_client_raises_without_key(self):
        from pybnt.ai.llm import _get_opencode_client

        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENCODEGO_API_KEY not set"):
                _get_opencode_client()

    def test_ask_llm_openode_handles_import_error(self):
        """Falls back to None when openai package is missing."""
        with mock.patch.dict(os.environ, {"OPENCODEGO_API_KEY": "k"}):
            with mock.patch("pybnt.ai.llm._get_opencode_client",
                            side_effect=ImportError("no openai")):
                result = ask_llm_openode(
                    [{"role": "user", "content": "hi"}]
                )
                assert result is None


class TestFallbackToDashScope:
    """DashScope provider still works when explicitly selected."""

    def test_ask_llm_dashscope_success(self):
        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.output.text = "Hello from Qwen"

        fake_dashscope = mock.MagicMock()
        fake_dashscope.Generation.call.return_value = fake_response

        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "ds-key"}):
            with mock.patch.dict(sys.modules, {"dashscope": fake_dashscope}):
                result = ask_llm_dashscope(
                    [{"role": "user", "content": "hi"}]
                )
                assert result == "Hello from Qwen"
                fake_dashscope.Generation.call.assert_called_once()

    def test_ask_llm_dashscope_no_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            result = ask_llm_dashscope(
                [{"role": "user", "content": "hi"}]
            )
            assert result is None

    def test_ask_llm_dashscope_error(self):
        fake_response = mock.MagicMock()
        fake_response.status_code = 400
        fake_response.message = "Bad request"

        fake_dashscope = mock.MagicMock()
        fake_dashscope.Generation.call.return_value = fake_response

        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}):
            with mock.patch.dict(sys.modules, {"dashscope": fake_dashscope}):
                result = ask_llm_dashscope(
                    [{"role": "user", "content": "hi"}]
                )
                assert result is None

    def test_ask_llm_routes_to_dashscope_with_provider_arg(self):
        """ask_llm with provider='dashscope' calls dashscope path."""
        with mock.patch("pybnt.ai.llm.ask_llm_dashscope") as mock_ds:
            mock_ds.return_value = "dashscope response"
            result = ask_llm(
                [{"role": "user", "content": "hi"}],
                provider="dashscope",
            )
            assert result == "dashscope response"
            mock_ds.assert_called_once()


class TestBackwardCompatCallAlias:
    """call = ask_llm alias is preserved."""

    def test_call_is_ask_llm(self):
        assert call is ask_llm

    def test_call_can_be_used(self):
        with mock.patch("pybnt.ai.llm.ask_llm_openode") as mock_oc:
            mock_oc.return_value = "ok"
            result = call([{"role": "user", "content": "hi"}])
            assert result == "ok"

    def test_ask_llm_sets_api_key_for_opencodego(self):
        """Backward-compat api_key param sets env var for opencodego."""
        with mock.patch("pybnt.ai.llm.ask_llm_openode") as mock_oc:
            mock_oc.return_value = "ok"
            result = ask_llm(
                [{"role": "user", "content": "hi"}],
                api_key="my-legacy-key",
            )
            assert result == "ok"
            assert os.environ.get("OPENCODEGO_API_KEY") == "my-legacy-key"

    def test_ask_llm_sets_api_key_for_dashscope(self):
        """Backward-compat api_key param sets env var for dashscope."""
        with mock.patch("pybnt.ai.llm.ask_llm_dashscope") as mock_ds:
            mock_ds.return_value = "ok"
            result = ask_llm(
                [{"role": "user", "content": "hi"}],
                api_key="my-ds-key",
                provider="dashscope",
            )
            assert result == "ok"
            assert os.environ.get("DASHSCOPE_API_KEY") == "my-ds-key"


class TestMultimodalPreparesCorrectly:
    """ask_llm_multimodal constructs correct base64 image messages."""

    def test_multimodal_builds_image_message(self, tmp_path):
        img = tmp_path / "brain.png"
        img.write_bytes(b"\x89PNG fake image data")

        with mock.patch("pybnt.ai.llm.ask_llm_openode") as mock_oc:
            mock_oc.return_value = "brain scan analysis"
            result = ask_llm_multimodal(
                str(img),
                prompt="Analyze this scan",
            )
            assert result == "brain scan analysis"
            call_args = mock_oc.call_args
            messages = call_args[0][0]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            content = messages[0]["content"]
            assert content[0]["type"] == "text"
            assert content[0]["text"] == "Analyze this scan"
            assert content[1]["type"] == "image_url"
            assert "data:image/png;base64," in content[1]["image_url"]["url"]

    def test_multimodal_default_model_mimo(self, tmp_path):
        img = tmp_path / "scan.jpg"
        img.write_bytes(b"\xff\xd8 fake jpeg")

        with mock.patch("pybnt.ai.llm.ask_llm_openode") as mock_oc:
            mock_oc.return_value = "ok"
            ask_llm_multimodal(str(img))
            assert mock_oc.call_args[0][1] == "mimo-v2.5"

    def test_multimodal_dashscope_warns(self, tmp_path):
        img = tmp_path / "x.png"
        img.write_bytes(b"fake")

        result = ask_llm_multimodal(
            str(img), provider="dashscope"
        )
        assert result is None

    def test_multimodal_detects_mime_from_extension(self, tmp_path):
        img = tmp_path / "scan.jpeg"
        img.write_bytes(b"\xff\xd8 fake")

        with mock.patch("pybnt.ai.llm.ask_llm_openode") as mock_oc:
            mock_oc.return_value = "ok"
            ask_llm_multimodal(str(img))
            messages = mock_oc.call_args[0][0]
            url = messages[0]["content"][1]["image_url"]["url"]
            assert "data:image/jpeg;base64," in url


class TestInvalidProviderReturnsNone:
    """Unknown provider returns None with error log."""

    def test_invalid_provider_returns_none(self):
        result = ask_llm(
            [{"role": "user", "content": "hi"}],
            provider="openai",  # type: ignore[arg-type]
        )
        assert result is None
