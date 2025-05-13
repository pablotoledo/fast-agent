from openai import AzureOpenAI, AuthenticationError, OpenAI
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.provider_types import Provider

try:
    from azure.identity import DefaultAzureCredential
except ImportError:
    DefaultAzureCredential = None

def _extract_resource_name(url: str) -> str | None:
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    suffix = ".openai.azure.com"
    return host.replace(suffix, "") if host.endswith(suffix) else None

DEFAULT_AZURE_API_VERSION = "2023-05-15"

class AzureOpenAIAugmentedLLM(OpenAIAugmentedLLM):
    """
    Azure OpenAI implementation extending OpenAIAugmentedLLM.
    Handles both API Key and DefaultAzureCredential authentication.
    """

    def __init__(self, provider: Provider = Provider.AZURE, *args, **kwargs):
        # Set provider to AZURE, pass through to base
        super().__init__(provider=provider, *args, **kwargs)

    def _openai_client(self) -> OpenAI:
        """
        Returns an AzureOpenAI client, handling both API Key and DefaultAzureCredential.
        """
        # Context/config extraction
        context = getattr(self, "context", None)
        config = getattr(context, "config", None) if context else None
        azure_cfg = getattr(config, "azure", None) if config else None

        if azure_cfg is None:
            raise ProviderKeyError(
                "Missing Azure configuration",
                "Azure provider requires configuration section 'azure' in your config file.",
            )

        use_default_cred = getattr(azure_cfg, "use_default_azure_credential", False)
        deployment_name = getattr(self, "default_request_params", None)
        deployment_name = getattr(deployment_name, "model", None) or getattr(azure_cfg, "azure_deployment", None)
        api_version = getattr(azure_cfg, "api_version", None) or DEFAULT_AZURE_API_VERSION

        if use_default_cred:
            base_url = getattr(azure_cfg, "base_url", None)
            if not base_url:
                raise ProviderKeyError(
                    "Missing Azure endpoint",
                    "When using 'use_default_azure_credential', 'base_url' is required in azure config.",
                )
            if DefaultAzureCredential is None:
                raise ProviderKeyError(
                    "azure-identity not installed",
                    "You must install 'azure-identity' to use DefaultAzureCredential authentication.",
                )
            credential = DefaultAzureCredential()
            def get_azure_token():
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                return token.token
            try:
                return AzureOpenAI(
                    azure_ad_token_provider=get_azure_token,
                    azure_endpoint=base_url,
                    api_version=api_version,
                    azure_deployment=deployment_name,
                )
            except AuthenticationError as e:
                raise ProviderKeyError(
                    "Invalid Azure AD credentials",
                    "The configured Azure AD credentials were rejected.\n"
                    "Please check your Azure identity setup.",
                ) from e
        else:
            api_key = getattr(azure_cfg, "api_key", None)
            resource_name = getattr(azure_cfg, "resource_name", None)
            base_url = getattr(azure_cfg, "base_url", None) or (f"https://{resource_name}.openai.azure.com/" if resource_name else None)
            if not api_key:
                raise ProviderKeyError(
                    "Missing Azure OpenAI credentials",
                    "Field 'api_key' is required in azure config.",
                )
            if not (resource_name or base_url):
                raise ProviderKeyError(
                    "Missing Azure endpoint",
                    "Provide either 'resource_name' or 'base_url' under azure config.",
                )
            if not deployment_name:
                raise ProviderKeyError(
                    "Missing deployment name",
                    "Set 'azure_deployment' in config or pass model=<deployment>.",
                )
            # If resource_name was missing, try to extract it from base_url
            if not resource_name and base_url:
                resource_name = _extract_resource_name(base_url)
            try:
                return AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=api_version,
                    azure_deployment=deployment_name,
                )
            except AuthenticationError as e:
                raise ProviderKeyError(
                    "Invalid Azure OpenAI API key",
                    "The configured Azure OpenAI API key was rejected.\n"
                    "Please check that your API key is valid and not expired.",
                ) from e
