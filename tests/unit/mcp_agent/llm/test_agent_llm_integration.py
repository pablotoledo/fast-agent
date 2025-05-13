import types

import pytest

from mcp_agent.llm.augmented_llm import RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_azure import AzureOpenAIAugmentedLLM


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
async def test_agent_returns_llm_response_default_azure_credential(monkeypatch):
    """
    Integration test for AzureOpenAIAugmentedLLM with use_default_azure_credential: True.
    Mocks DefaultAzureCredential and AzureOpenAI to ensure correct integration.
    """

    # Dummy token and credential
    class DummyToken:
        def __init__(self, token):
            self.token = token

    class DummyCredential:
        def get_token(self, scope):
            assert scope == "https://cognitiveservices.azure.com/.default"
            return DummyToken("dummy-token")

    # Patch DefaultAzureCredential to return DummyCredential
    import mcp_agent.llm.providers.augmented_llm_azure as azure_mod

    monkeypatch.setattr(azure_mod, "DefaultAzureCredential", DummyCredential)

    # Patch AzureOpenAI to check for azure_ad_token_provider and simulate response
    class DummyAzureOpenAI:
        def __init__(self, **kwargs):
            assert "azure_ad_token_provider" in kwargs
            self.token_provider = kwargs["azure_ad_token_provider"]
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kw: types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(content="tokenpong")
                            )
                        ]
                    )
                )
            )

    monkeypatch.setattr(azure_mod, "AzureOpenAI", DummyAzureOpenAI)

    # Prepare config for DefaultAzureCredential
    class DACfg:
        api_key = None
        resource_name = None
        azure_deployment = "test-deployment"
        api_version = "2023-05-15"
        base_url = "https://mydemo.openai.azure.com/"
        use_default_azure_credential = True

    class DummyConfigDAC:
        azure = DACfg()
        logger = DummyLogger()

    class DummyContextDAC:
        config = DummyConfigDAC()
        executor = None

    llm = AzureOpenAIAugmentedLLM(
        provider=Provider.AZURE, context=DummyContextDAC(), model="test-deployment"
    )

    # The token provider should return the dummy token
    assert llm.client.token_provider() == "dummy-token"

    # Simulate the agent using the LLM provider
    messages = [{"role": "user", "content": "hola"}]
    params = RequestParams()
    response = await llm._chat_completion(messages, params)
    assert response.choices[0].message.content == "tokenpong"

    # Simulate the higher-level agent returning the LLM result to the user
    from mcp.types import TextContent

    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

    user_msg = PromptMessageMultipart(role="user", content=[TextContent(type="text", text="hola")])
    result = await llm._apply_prompt_provider_specific([user_msg], params)
    assert result.last_text() == "tokenpong"
    assert result.role == "assistant"


@pytest.mark.asyncio
async def test_agent_returns_llm_response(monkeypatch):
    # Patch the AzureOpenAI client method on the instance (sync monkey-patch)
    def fake_create(model, messages, **kw):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="pong"))]
        )

    # Instantiate the provider and patch the client
    llm = AzureOpenAIAugmentedLLM(
        provider=Provider.AZURE, context=DummyContext(), model="test-deployment"
    )
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    # Simulate the agent using the LLM provider
    # For this test, we simulate a user prompt and check the agent's response
    messages = [{"role": "user", "content": "hola"}]
    params = RequestParams()
    response = await llm._chat_completion(messages, params)
    assert response.choices[0].message.content == "pong"

    # Simulate the higher-level agent returning the LLM result to the user
    from mcp.types import TextContent

    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

    user_msg = PromptMessageMultipart(role="user", content=[TextContent(type="text", text="hola")])
    result = await llm._apply_prompt_provider_specific([user_msg], params)
    assert result.last_text() == "pong"
    assert result.role == "assistant"
