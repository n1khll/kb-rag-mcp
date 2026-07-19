from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Postgres with pgvector (Neon or any public Postgres). Include ?sslmode=require for Neon.
    database_url: str = Field(alias="DATABASE_URL")

    # Azure AD service principal used to call Azure OpenAI embeddings.
    azure_tenant_id: str = Field(alias="AZURE_TENANT_ID")
    azure_client_id: str = Field(alias="AZURE_CLIENT_ID")
    azure_client_secret: str = Field(alias="AZURE_CLIENT_SECRET")

    # Bare Azure AI resource endpoint, e.g. https://my-resource.services.ai.azure.com/
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_embed_deployment: str = Field(default="text-embedding-3-small", alias="AZURE_OPENAI_EMBED_DEPLOYMENT")
    azure_openai_api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")

    # Comma-separated list of accepted bearer tokens for MCP clients.
    # Empty list = server refuses all tool traffic (fail closed, not open).
    mcp_api_keys: str = Field(default="", alias="MCP_API_KEYS")

    # Extra Host header values to accept (DNS-rebinding allow-list), comma-separated.
    # Use "*" to disable the check entirely. The Render deployment host is added
    # automatically from RENDER_EXTERNAL_HOSTNAME, so this is only needed for custom
    # domains or other hosting platforms.
    mcp_allowed_hosts: str = Field(default="", alias="MCP_ALLOWED_HOSTS")
    render_external_hostname: str = Field(default="", alias="RENDER_EXTERNAL_HOSTNAME")

    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")

    @property
    def api_key_set(self) -> frozenset[str]:
        return frozenset(k.strip() for k in self.mcp_api_keys.split(",") if k.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
