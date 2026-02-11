"""Google implementation of embedder."""

from __future__ import annotations

import openai
from django.conf import settings

from apps.ai.embeddings.base import Embedder


class GoogleEmbedder(Embedder):
    """Google implementation of embedder using OpenAI compatible endpoint."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize Google embedder.

        Args:
            model: The Google embedding model to use. If None, uses settings.

        """
        self.api_key = settings.GOOGLE_API_KEY
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=30,
        )
        self.model = model or settings.GOOGLE_EMBEDDING_MODEL_NAME
        self._dimensions = 768  # text-embedding-004 dimensions

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query string.

        Args:
            text: The query text to embed.

        Returns:
            List of floats representing the embedding vector.

        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one per document.

        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def get_dimensions(self) -> int:
        """Get the dimension of embeddings produced by this embedder.

        Returns:
            Integer representing the embedding dimension.

        """
        return self._dimensions
