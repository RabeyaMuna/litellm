import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


def _get_base_validate_environment_globals():
    base_azure_llm = AzureOpenAIResponsesAPIConfig.validate_environment.__globals__[
        "BaseAzureLLM"
    ]
    return base_azure_llm._base_validate_azure_environment.__globals__


def test_validate_environment_api_key_within_litellm_params():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    litellm_params = GenericLiteLLMParams(api_key="test-api-key")

    result = azure_openai_responses_apiconfig.validate_environment(
        headers={}, model="", litellm_params=litellm_params
    )

    expected = {"api-key": "test-api-key"}

    assert result == expected


def test_validate_environment_api_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    base_validate_globals = _get_base_validate_environment_globals()
    config_litellm = base_validate_globals["litellm"]

    with patch.object(config_litellm, "api_key", "test-api-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-api-key"}

        assert result == expected


def test_validate_environment_azure_key_within_litellm():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    base_validate_globals = _get_base_validate_environment_globals()
    config_litellm = base_validate_globals["litellm"]

    with patch.object(config_litellm, "azure_key", "test-azure-key"):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-azure-key"}

        assert result == expected


def test_validate_environment_azure_openai_api_key_within_secret_str():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    base_validate_globals = _get_base_validate_environment_globals()
    config_litellm = base_validate_globals["litellm"]

    def mock_get_secret_str(key):
        # Configure the mock to return "test-api-key" when called with "AZURE_OPENAI_API_KEY"
        return "test-api-key" if key == "AZURE_OPENAI_API_KEY" else None

    with (
        patch.object(config_litellm, "api_key", None),
        patch.object(config_litellm, "azure_key", None),
        patch.dict(
            base_validate_globals,
            {"get_secret_str": mock_get_secret_str},
        ),
    ):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )

        expected = {"api-key": "test-api-key"}

        assert result == expected


def test_validate_environment_azure_api_key_within_secret_str():
    azure_openai_responses_apiconfig = AzureOpenAIResponsesAPIConfig()
    base_validate_globals = _get_base_validate_environment_globals()
    config_litellm = base_validate_globals["litellm"]

    def mock_get_secret_str(key):
        # Return None for AZURE_OPENAI_API_KEY and a value for AZURE_API_KEY.
        if key == "AZURE_API_KEY":
            return "test-api-key"
        return None

    with (
        patch.object(config_litellm, "api_key", None),
        patch.object(config_litellm, "azure_key", None),
        patch.dict(
            base_validate_globals,
            {"get_secret_str": mock_get_secret_str},
        ),
    ):
        litellm_params = GenericLiteLLMParams()
        result = azure_openai_responses_apiconfig.validate_environment(
            headers={}, model="", litellm_params=litellm_params
        )
        expected = {"api-key": "test-api-key"}

        assert result == expected
