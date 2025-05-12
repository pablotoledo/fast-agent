import types
import pytest
import asyncio

from mcp_agent.llm.providers.augmented_llm_azure import AzureOpenAIAugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.augmented_llm import RequestParams
from mcp_agent.core.agent_app import AgentApp
from mcp_agent.core.prompt import Prompt

class DummyLogger:
    enable_markup = True

class DummyAzureConfig:
    api_key = "test-key"
    resource_name = "test-resource"
    azure_deployment = "test-deployment"
    api_version = "2023-05-15"
    base_url = None

class DummyConfig:
    azure = DummyAzureConfig()
    logger = DummyLogger()

class DummyContext:
    config = DummyConfig()
    executor = None

@pytest.mark.asyncio
async def test_agent_returns_llm_response(monkeypatch):
    # Patch the AzureOpenAI client method on the instance (sync monkey-patch)
    def fake_create(model, messages, **kw):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="pong")
        )])

    # Instantiate the provider and patch the client
    llm = AzureOpenAIAugmentedLLM(
        provider=Provider.AZURE,
        context=DummyContext(),
        model="test-deployment"
    )
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    # Simulate the agent using the LLM provider
    # For this test, we simulate a user prompt and check the agent's response
    messages = [{"role": "user", "content": "hola"}]
    params = RequestParams()
    response = await llm._chat_completion(messages, params)
    assert response.choices[0].message.content == "pong"

    # Simulate the higher-level agent returning the LLM result to the user
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
    from mcp.types import TextContent

    user_msg = PromptMessageMultipart(role="user", content=[TextContent(type="text", text="hola")])
    result = await llm._apply_prompt_provider_specific([user_msg], params)
    assert result.last_text() == "pong"
    assert result.role == "assistant"
