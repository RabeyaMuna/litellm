import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))
import json

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.openai import (
    IncompleteDetails,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIResponse,
    ResponseTextConfig,
)
from litellm.types.utils import StandardLoggingPayload


@pytest.mark.asyncio
async def test_responses_api_routing_with_previous_response_id():
    """
    Test that when using a previous_response_id, the request is sent to the same model_id
    """
    # Create a mock response that simulates Azure responses API
    mock_response_id = "resp_mock-resp-456"

    mock_response_data = {
        "id": mock_response_id,
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "I'm doing well, thank you for asking!",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    # Mock all endpoints
    respx.post(url__regex=r"https://mock-endpoint.*\.openai\.azure\.com/.*").mock(
        return_value=respx.MockResponse(200, json=mock_response_data)
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version-2",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["responses_api_deployment_check"],
    )
    MODEL = "azure-computer-use-preview"

    # Make the initial request
    # litellm._turn_on_debug()
    response = await router.aresponses(
        model=MODEL,
        input="Hello, how are you?",
        truncation="auto",
    )
    print("RESPONSE", response)

    # Store the model_id from the response
    expected_model_id = response._hidden_params["model_id"]
    response_id = response.id

    print("Response ID=", response_id, "came from model_id=", expected_model_id)

    # Make 10 other requests with previous_response_id, assert that they are sent to the same model_id
    for i in range(10):
        response = await router.aresponses(
            model=MODEL,
            input=f"Follow-up question {i+1}",
            truncation="auto",
            previous_response_id=response_id,
        )

        # Assert the model_id is preserved
        assert response._hidden_params["model_id"] == expected_model_id


@pytest.mark.asyncio
async def test_routing_without_previous_response_id():
    """
    Test that normal routing (load balancing) works when no previous_response_id is provided
    """
    mock_response_data = {
        "id": "mock-resp-123",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello there!", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    # Create mock aiohttp response
    mock_aiohttp_response = AsyncMock()
    mock_aiohttp_response.status = 200
    mock_aiohttp_response.json = AsyncMock(return_value=mock_response_data)
    mock_aiohttp_response.text = json.dumps(mock_response_data)
    mock_aiohttp_response.__aenter__ = AsyncMock(return_value=mock_aiohttp_response)
    mock_aiohttp_response.__aexit__ = AsyncMock(return_value=None)

    # Create a router with two identical deployments to test load balancing
    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key-3",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-3.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key-4",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-4.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["responses_api_deployment_check"],
    )

    MODEL = "azure-computer-use-preview"

    # Mock aiohttp client session request
    with patch("aiohttp.ClientSession.request", return_value=mock_aiohttp_response):
        # Make multiple requests and verify we're hitting different deployments
        used_model_ids = set()

        for i in range(20):
            response = await router.aresponses(
                model=MODEL,
                input=f"Question {i}",
                truncation="auto",
            )

            used_model_ids.add(response._hidden_params["model_id"])

        # We should have used more than one model_id if load balancing is working
        assert (
            len(used_model_ids) > 1
        ), "Load balancing isn't working, only one deployment was used"


@pytest.mark.asyncio
@respx.mock
async def test_previous_response_id_not_in_cache():
    """
    Test behavior when a previous_response_id is provided but not found in cache
    """
    mock_response_data = {
        "id": "mock-resp-789",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Nice to meet you!",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    # Mock all endpoints
    respx.post(url__regex=r"https://mock-endpoint-\d+\.openai\.azure\.com/.*").mock(
        return_value=respx.MockResponse(200, json=mock_response_data)
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["responses_api_deployment_check"],
    )

    MODEL = "azure-computer-use-preview"

    # Make a request with a non-existent previous_response_id
    response = await router.aresponses(
        model=MODEL,
        input="Hello, this is a test",
        truncation="auto",
        previous_response_id="non-existent-response-id",
    )

    # Should still get a valid response
    assert response is not None
    assert response.id is not None

    # Since the previous_response_id wasn't found, routing should work normally
    # We can't assert exactly which deployment was chosen, but we can verify the basics
    assert response._hidden_params["model_id"] is not None


@pytest.mark.asyncio
@respx.mock
async def test_multiple_response_ids_routing():
    """
    Test that different response IDs correctly route to their respective original deployments
    """
    # Create two different mock responses for our two different deployments
    mock_response_data_1 = {
        "id": "mock-resp-deployment-1",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Response from deployment 1",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    mock_response_data_2 = {
        "id": "mock-resp-deployment-2",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "model": "azure/computer-use-preview",
        "output": [
            {
                "type": "message",
                "id": "msg_456",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Response from deployment 2",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "truncation": "disabled",
        "user": None,
    }

    # Use side_effect to return different responses
    call_count = [0]

    def get_response(*args, **kwargs):
        call_count[0] += 1
        # Alternate between the two responses
        if call_count[0] % 2 == 1:
            return respx.MockResponse(200, json=mock_response_data_1)
        else:
            return respx.MockResponse(200, json=mock_response_data_2)

    # Mock all endpoints with side_effect
    respx.post(url__regex=r"https://mock-endpoint-\d+\.openai\.azure\.com/.*").mock(
        side_effect=get_response
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-1",
                    "api_key": "mock-api-key-1",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-1.openai.azure.com",
                },
            },
            {
                "model_name": "azure-computer-use-preview",
                "litellm_params": {
                    "model": "azure/computer-use-preview-2",
                    "api_key": "mock-api-key-2",
                    "api_version": "mock-api-version",
                    "api_base": "https://mock-endpoint-2.openai.azure.com",
                },
            },
        ],
        optional_pre_call_checks=["responses_api_deployment_check"],
    )

    MODEL = "azure-computer-use-preview"

    # Make the first request to deployment 1
    response1 = await router.aresponses(
        model=MODEL,
        input="Request to deployment 1",
        truncation="auto",
    )

    # Store details from first response
    model_id_1 = response1._hidden_params["model_id"]
    response_id_1 = response1.id

    # Make the second request to deployment 2
    response2 = await router.aresponses(
        model=MODEL,
        input="Request to deployment 2",
        truncation="auto",
    )

    # Store details from second response
    model_id_2 = response2._hidden_params["model_id"]
    response_id_2 = response2.id

    # Wait for cache updates
    await asyncio.sleep(1)

    # Now make follow-up requests using the previous response IDs

    # Follow-up to response 1 should go to model_id_1
    follow_up_1 = await router.aresponses(
        model=MODEL,
        input="Follow up to deployment 1",
        truncation="auto",
        previous_response_id=response_id_1,
    )

    # Verify it went to the correct deployment
    assert follow_up_1._hidden_params["model_id"] == model_id_1

    # Follow-up to response 2 should go to model_id_2
    follow_up_2 = await router.aresponses(
        model=MODEL,
        input="Follow up to deployment 2",
        truncation="auto",
        previous_response_id=response_id_2,
    )

    # Verify it went to the correct deployment
    assert follow_up_2._hidden_params["model_id"] == model_id_2
