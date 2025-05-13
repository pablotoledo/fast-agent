import asyncio
from typing import Any, Dict, List
from urllib.parse import urlparse

from openai import AzureOpenAI

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt

try:
    from azure.identity import DefaultAzureCredential
except ImportError:
    DefaultAzureCredential = None
from mcp_agent.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.llm.provider_types import Provider
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


def _extract_resource_name(url: str) -> str | None:
    """
    Dada una URL https://<resource>.openai.azure.com/...
    devuelve '<resource>' o None si no encaja.
    """
    host = urlparse(url).hostname or ""
    suffix = ".openai.azure.com"
    return host.replace(suffix, "") if host.endswith(suffix) else None


DEFAULT_AZURE_API_VERSION = "2023-05-15"


def _multipart_to_openai(msg):
    parts = [c.text for c in getattr(msg, "content", []) if getattr(c, "type", "text") == "text"]
    return {"role": msg.role, "content": "".join(parts)}


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
                "Azure provider requires configuration section 'azure' in your config file.",
            )

        # --- Retrocompatible Auth: API Key o DefaultAzureCredential ---
        use_default_cred = getattr(azure_cfg, "use_default_azure_credential", False)
        self.deployment_name = kwargs.get("model") or azure_cfg.azure_deployment
        self.api_version = azure_cfg.api_version or DEFAULT_AZURE_API_VERSION

        if use_default_cred:
            # Modo DefaultAzureCredential
            if not azure_cfg.base_url:
                raise ProviderKeyError(
                    "Missing Azure endpoint",
                    "When using 'use_default_azure_credential', 'base_url' is required in azure config.",
                )
            if DefaultAzureCredential is None:
                raise ProviderKeyError(
                    "azure-identity not installed",
                    "You must install 'azure-identity' to use DefaultAzureCredential authentication.",
                )
            self.base_url = azure_cfg.base_url
            self.api_key = None  # No usar api_key
            self.resource_name = None
            # Token provider callable
            credential = DefaultAzureCredential()
            def get_azure_token():
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                return token.token
            self.logger.info(
                f"AzureOpenAI endpoint: {self.base_url} — deployment: {self.deployment_name} (DefaultAzureCredential)"
            )
            self.client = AzureOpenAI(
                azure_ad_token_provider=get_azure_token,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                azure_deployment=self.deployment_name,
            )
        else:
            # Modo API Key (actual)
            self.api_key = azure_cfg.api_key
            self.resource_name = azure_cfg.resource_name
            if not self.api_key:
                raise ProviderKeyError(
                    "Missing Azure OpenAI credentials", "Field 'api_key' is required in azure config."
                )
            if not (self.resource_name or azure_cfg.base_url):
                raise ProviderKeyError(
                    "Missing Azure endpoint",
                    "Provide either 'resource_name' or 'base_url' under azure config.",
                )
            if not self.deployment_name:
                raise ProviderKeyError(
                    "Missing deployment name",
                    "Set 'azure_deployment' in config or pass model=<deployment>.",
                )
            self.base_url = azure_cfg.base_url or f"https://{self.resource_name}.openai.azure.com/"
            # Si resource_name faltaba intenta extraerlo de base_url
            if not self.resource_name and azure_cfg.base_url:
                self.resource_name = _extract_resource_name(azure_cfg.base_url)
            self.logger.info(
                f"AzureOpenAI endpoint: {self.base_url} — deployment: {self.deployment_name}"
            )
            self.client = AzureOpenAI(
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
        kwargs = {
            "model": self.deployment_name,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if request_params.maxTokens is not None:
            kwargs["max_tokens"] = request_params.maxTokens

        self.logger.debug(f"Azure OpenAI completion requested for: {kwargs}")

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **kwargs,
            )
            self.logger.debug(f"Azure response: {response}")
        except Exception as e:
            self.logger.error(f"Azure OpenAI API error: {e}")
            raise ProviderKeyError("Azure OpenAI API error", f"Error from Azure OpenAI: {e}") from e

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
        # Convert messages to OpenAI format using _multipart_to_openai
        converted = [_multipart_to_openai(m) for m in messages_to_add]

        self.history.extend(converted, is_prompt=is_template)

        if "assistant" == last_message.role:
            return last_message

        # For assistant messages: Return the last message (no completion needed)
        message_param = _multipart_to_openai(last_message)
        request_params = self.get_request_params(request_params=request_params)
        messages = converted + [message_param]

        response = await self._chat_completion(messages, request_params)
        if not getattr(response, "choices", None):
            self.logger.error(f"Azure returned no choices: {response}")
            return Prompt.assistant("[empty]")

        choice = response.choices[0]
        content = getattr(choice.message, "content", None) or ""
        return Prompt.assistant(content)
