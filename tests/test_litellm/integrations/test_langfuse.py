import json
import os
import sys
from types import SimpleNamespace
from typing import Optional

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

import asyncio
from unittest.mock import patch

import pytest

import litellm
from litellm.integrations.langfuse.langfuse import LangFuseLogger


def test_max_langfuse_clients_limit(monkeypatch):
    """
    Test that the max langfuse clients limit is respected when initializing multiple clients
    """
    class MockLangfuse:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.client = SimpleNamespace(
                projects=SimpleNamespace(
                    get=lambda: SimpleNamespace(
                        data=[SimpleNamespace(id="test-project-id")]
                    )
                )
            )

    mock_langfuse_module = SimpleNamespace(
        Langfuse=MockLangfuse,
        version=SimpleNamespace(__version__="2.6.0"),
    )
    monkeypatch.setitem(sys.modules, "langfuse", mock_langfuse_module)

    # Set max clients to 2 for testing
    with patch(
        "litellm.integrations.langfuse.langfuse.MAX_LANGFUSE_INITIALIZED_CLIENTS", 2
    ):
        # Reset the counter
        monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)

        # First client should succeed
        LangFuseLogger(
            langfuse_public_key="test_key_1",
            langfuse_secret="test_secret_1",
            langfuse_host="https://test1.langfuse.com",
        )
        assert litellm.initialized_langfuse_clients == 1

        # Second client should succeed
        LangFuseLogger(
            langfuse_public_key="test_key_2",
            langfuse_secret="test_secret_2",
            langfuse_host="https://test2.langfuse.com",
        )
        assert litellm.initialized_langfuse_clients == 2

        # Third client should fail with exception
        with pytest.raises(Exception) as exc_info:
            LangFuseLogger(
                langfuse_public_key="test_key_3",
                langfuse_secret="test_secret_3",
                langfuse_host="https://test3.langfuse.com",
            )

        # Verify the error message contains the expected text
        assert "Max langfuse clients reached" in str(exc_info.value)

        # Counter should still be 2 (third client failed to initialize)
        assert litellm.initialized_langfuse_clients == 2
