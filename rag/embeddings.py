import asyncio
from typing import List, Optional

from azure.identity import ClientSecretCredential, get_bearer_token_provider
from openai import AzureOpenAI

from rag.config import Settings


class EmbeddingService:
    """Azure OpenAI embeddings authenticated via an explicit service principal.

    Explicit ClientSecretCredential (not DefaultAzureCredential) so behavior is identical
    on a laptop, in CI, and on Render -- no dependency on ambient host identity.
    """

    def __init__(self, settings: Settings):
        credential = ClientSecretCredential(
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
        )
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_embed_deployment

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self._deployment, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)


_service: Optional[EmbeddingService] = None


def get_embedding_service(settings: Settings) -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService(settings)
    return _service
