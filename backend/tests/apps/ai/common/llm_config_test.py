from unittest.mock import patch

from django.test import SimpleTestCase

from apps.ai.common.llm_config import get_llm


class LLMConfigTest(SimpleTestCase):
    """Test LLM configuration and selection."""

    @patch("apps.ai.common.llm_config.settings")
    def test_get_llm_openai(self, mock_settings):
        """Test that get_llm returns OpenAI LLM by default or when configured."""
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.OPENAI_MODEL_NAME = "gpt-4o"
        mock_settings.OPEN_AI_SECRET_KEY = "sk-test"

        llm = get_llm()

        self.assertEqual(llm.model, "gpt-4o")
        self.assertEqual(llm.api_key, "sk-test")

    @patch("apps.ai.common.llm_config.settings")
    def test_get_llm_gemini(self, mock_settings):
        """Test that get_llm returns Gemini LLM when configured."""
        mock_settings.LLM_PROVIDER = "google"
        mock_settings.GOOGLE_MODEL_NAME = "gemini-1.5-flash"
        mock_settings.GOOGLE_API_KEY = "gemini-test"

        llm = get_llm()

        # CrewAI LLM (LiteLLM) expects openai/ prefix for custom base_urls
        self.assertEqual(llm.model, "openai/gemini-1.5-flash")
        self.assertEqual(llm.api_key, "gemini-test")
        self.assertEqual(llm.base_url, "https://generativelanguage.googleapis.com/v1beta/openai/")

    @patch("apps.ai.common.llm_config.settings")
    def test_get_llm_fallback(self, mock_settings):
        """Test that get_llm falls back to OpenAI for unknown providers."""
        mock_settings.LLM_PROVIDER = "unknown"
        mock_settings.OPENAI_MODEL_NAME = "gpt-4o-mini"
        mock_settings.OPEN_AI_SECRET_KEY = "sk-fallback"

        llm = get_llm()

        self.assertEqual(llm.model, "gpt-4o-mini")
        self.assertEqual(llm.api_key, "sk-fallback")
