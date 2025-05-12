import asyncio
from typing import List, Dict, Any

from openai import AsyncAzureOpenAI
from mcp_agent.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

DEFAULT_AZURE_API_VERSION = "2023-05-15"

class AzureOpenAIAugmentedLLM(AugmentedLLM):
    """
    AugmentedLLM implementation for Azure OpenAI Service.
    Configures the openai SDK for Azure and uses deployment_name as engine.
    """

    def __init__(self, provider: Provider = Provider.AZURE, *args, **kwargs):
        # Set up logger
        self.logger = get_logger(f"{__name__}.{self.name}" if hasattr(self, "name") else __name__)

        # Extract context/config
        context = kwargs.get("context", None)
        config = None
        if context and hasattr(context, "config"):
            config = context.config
        elif hasattr(self, "context") and hasattr(self.context, "config"):
            config = self.context.config

        azure_cfg = None
        if config and hasattr(config, "azure"):
            azure_cfg = config.azure

        if azure_cfg is None:
            raise ProviderKeyError(
                "Missing Azure configuration",
                "Azure provider requires configuration section 'azure' in your config file."
            )

        # Set up Azure OpenAI parameters
        self.api_key = azure_cfg.api_key
        self.resource_name = azure_cfg.resource_name
        self.deployment_name = kwargs.get("model") or azure_cfg.azure_deployment
        self.api_version = azure_cfg.api_version or DEFAULT_AZURE_API_VERSION
        self.base_url = azure_cfg.base_url or (f"https://{self.resource_name}.openai.azure.com/")

        if not self.api_key or not self.resource_name or not self.deployment_name:
            raise ProviderKeyError(
                "Missing Azure OpenAI credentials",
                "Azure provider requires 'api_key', 'resource_name', and 'azure_deployment' in config."
            )

        # Set up AsyncAzureOpenAI client (per-instance, no global state)
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.base_url,
            api_version=self.api_version,
            azure_deployment=self.deployment_name,
        )

        # Set type_converter if needed (optional, for compatibility)
        if "type_converter" not in kwargs:
            kwargs["type_converter"] = None

        super().__init__(*args, provider=provider, **kwargs)

    async def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        request_params: RequestParams,
        tools=None,
    ):
        """
        Call Azure OpenAI ChatCompletion endpoint using deployment_name as model.
        """
        arguments = {
            "model": self.deployment_name,
            "messages": messages,
        }
        if tools is not None:
            arguments["tools"] = tools
        if request_params.maxTokens is not None:
            arguments["max_tokens"] = request_params.maxTokens

        self.logger.debug(f"Azure OpenAI completion requested for: {arguments}")

        try:
            response = await self.client.chat.completions.create(**arguments)
            self.logger.debug(f"Azure response: {response}")
        except Exception as e:
            self.logger.error(f"Azure OpenAI API error: {e}")
            raise ProviderKeyError(
                "Azure OpenAI API error",
                f"Error from Azure OpenAI: {e}"
            ) from e

        return response

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        # This mirrors the OpenAI logic, adapted for Azure
        last_message = multipart_messages[-1]
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        # Convert messages to OpenAI format (assume compatible)
        converted = []
        for msg in messages_to_add:
            # For simplicity, assume msg.to_dict() gives the right format
            # In a real implementation, use a converter as in OpenAI provider
            converted.append(msg.to_dict() if hasattr(msg, "to_dict") else dict(msg))

        self.history.extend(converted, is_prompt=is_template)

        if "assistant" == last_message.role:
            return last_message

        # For assistant messages: Return the last message (no completion needed)
        message_param = last_message.to_dict() if hasattr(last_message, "to_dict") else dict(last_message)
        request_params = self.get_request_params(request_params=request_params)
        messages = converted + [message_param]

        response = await self._chat_completion(messages, request_params)
        if not getattr(response, "choices", None):
            self.logger.error(f"Azure returned no choices: {response}")
            return Prompt.assistant("[empty]")

        choice = response.choices[0]
        content = getattr(choice.message, "content", None) or ""
        return Prompt.assistant(content)
