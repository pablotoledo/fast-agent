import types
import pytest
import asyncio

from mcp_agent.llm.providers.augmented_llm_azure import AzureOpenAIAugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.augmented_llm import RequestParams
from mcp.types import TextContent

class DummyLogger:
    enable_markup = True

class DummyAzureConfig:
    def __init__(self):
        self.api_key = "test-key"
        self.resource_name = "test-resource"
        self.azure_deployment = "test-deployment"
        self.api_version = "2023-05-15"
        self.base_url = None

class DummyConfig:
    def __init__(self):
        self.azure = DummyAzureConfig()
        self.logger = DummyLogger()

class DummyContext:
    def __init__(self):
        self.config = DummyConfig()
        self.executor = None

def test_init_with_base_url_only():
    cfg = DummyAzureConfig()
    cfg.base_url = "https://mydemo.openai.azure.com/"
    cfg.resource_name = None
    class Ctx:
        def __init__(self):
            self.config = type("Config", (), {"azure": cfg, "logger": DummyLogger()})()
            self.executor = None
    llm = AzureOpenAIAugmentedLLM(context=Ctx())
    assert llm.base_url.startswith("https://mydemo.openai.azure.com")
    assert llm.resource_name == "mydemo"

@pytest.mark.asyncio
async def test_azure_llm_chat_completion(monkeypatch):
    # Patch the AzureOpenAI client method on the instance (sync monkey-patch)
    def fake_create(model, messages, **kw):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="pong")
        )])
    # Instantiate the provider
    llm = AzureOpenAIAugmentedLLM(
        provider=Provider.AZURE,
        context=DummyContext(),
        model="test-deployment"
    )
    # Patch the instance's sync client method using monkeypatch to avoid assignment error
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    # Prepare a minimal prompt
    messages = [{"role": "user", "content": "ping"}]
    params = RequestParams()
    response = await llm._chat_completion(messages, params)
    assert response.choices[0].message.content == "pong"

    # Test the provider returns the expected assistant message
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
    user_msg = PromptMessageMultipart(role="user", content=[TextContent(type="text", text="ping")])
    result = await llm._apply_prompt_provider_specific([user_msg], params)
    assert result.last_text() == "pong"
    assert result.role == "assistant"

@pytest.mark.asyncio
async def test_azure_llm_empty_choices_returns_empty_guard(monkeypatch):
    # Patch the AzureOpenAI client method to return no choices
    def fake_create(model, messages, **kw):
        return types.SimpleNamespace(choices=[])
    llm = AzureOpenAIAugmentedLLM(
        provider=Provider.AZURE,
        context=DummyContext(),
        model="test-deployment"
    )
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
    user_msg = PromptMessageMultipart(role="user", content=[TextContent(type="text", text="ping")])
    params = RequestParams()
    result = await llm._apply_prompt_provider_specific([user_msg], params)
    assert result.last_text() == "[empty]"
    assert result.role == "assistant"

@pytest.mark.asyncio
async def test_azure_llm_choice_content_none_returns_empty_string(monkeypatch):
    # Patch the AzureOpenAI client method to return a choice with content=None
    def fake_create(model, messages, **kw):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=None)
        )])
    llm = AzureOpenAIAugmentedLLM(
        provider=Provider.AZURE,
        context=DummyContext(),
        model="test-deployment"
    )
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
    user_msg = PromptMessageMultipart(role="user", content=[TextContent(type="text", text="ping")])
    params = RequestParams()
    result = await llm._apply_prompt_provider_specific([user_msg], params)
    assert result.last_text() == ""
    assert result.role == "assistant"

@pytest.mark.asyncio
async def test_openai_payload_is_plain_string(monkeypatch):
    # Capture the messages sent to the Azure client
    captured = {}

    def fake_create(model, messages, **kw):
        captured["messages"] = messages
        # Return a dummy response
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="ok")
        )])

    llm = AzureOpenAIAugmentedLLM(
        provider=Provider.AZURE,
        context=DummyContext(),
        model="test-deployment"
    )
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
    from mcp.types import TextContent

    # Compose a multipart message with multiple text parts
    user_msg = PromptMessageMultipart(
        role="user",
        content=[
            TextContent(type="text", text="foo"),
            TextContent(type="text", text="bar"),
        ]
    )
    params = RequestParams()
    await llm._apply_prompt_provider_specific([user_msg], params)

    # Assert that all message 'content' fields are plain strings, not lists
    for msg in captured["messages"]:
        assert isinstance(msg["content"], str), f"content is not str: {type(msg['content'])}"
