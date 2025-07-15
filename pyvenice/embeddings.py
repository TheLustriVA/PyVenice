"""
Embeddings endpoint wrapper for Venice.ai API.
"""

from typing import Union, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator

from .client import BaseResource, VeniceClient
from .models import Models


class EmbeddingRequest(BaseModel):
    """Request model for creating embeddings."""

    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    dimensions: Optional[int] = Field(
        None, ge=1, description="Number of dimensions for output"
    )
    encoding_format: Literal["float", "base64"] = "float"

    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        if isinstance(v, str) and len(v) == 0:
            raise ValueError("Input string cannot be empty")
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Input array cannot be empty")
            if len(v) > 2048:
                raise ValueError("Input array cannot have more than 2048 elements")
        return v


class EmbeddingObject(BaseModel):
    """Single embedding object in response."""

    embedding: Union[List[float], str]  # List of floats or base64 string
    index: int
    object: Literal["embedding"] = "embedding"


class EmbeddingResponse(BaseModel):
    """Response from embeddings endpoint."""

    data: List[EmbeddingObject]
    model: str
    object: Literal["list"] = "list"
    usage: dict


class Embeddings(BaseResource):
    """
    Interface for Venice.ai embeddings endpoint.

    Creates embeddings for text inputs using embedding models.
    This is a beta feature accessible to Venice beta testers.

    NOTE: The embeddings endpoint is only available to users in the Venice.ai beta program.
    Integration tests for embeddings functionality may fail if the API key does not have
    beta access. The dimensions parameter behavior may need validation once beta access
    is available for testing.
    """

    def __init__(self, client: VeniceClient):
        super().__init__(client)
        self.models = Models(client)

    def create(
        self,
        input: Union[str, List[str], List[int], List[List[int]]],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingResponse:
        """
        Create embeddings for the given input.

        Args:
            input: Text string(s) or token arrays to embed.
            model: The embedding model to use.
            dimensions: Number of dimensions for the output embeddings.
            encoding_format: Format for the embeddings (float or base64).

        Returns:
            EmbeddingResponse containing the embeddings.
        """
        # Map model name if needed
        model = self.models.map_model_name(model)

        request = EmbeddingRequest(
            input=input,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )

        response = self.client.post(
            "/embeddings", request.model_dump(exclude_none=True)
        )
        return EmbeddingResponse(**response)

    async def create_async(
        self,
        input: Union[str, List[str], List[int], List[List[int]]],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        encoding_format: Literal["float", "base64"] = "float",
    ) -> EmbeddingResponse:
        """Async version of create()."""
        model = self.models.map_model_name(model)

        request = EmbeddingRequest(
            input=input,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )

        response = await self.client.post_async(
            "/embeddings", request.model_dump(exclude_none=True)
        )
        return EmbeddingResponse(**response)

    def create_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        encoding_format: Literal["float", "base64"] = "float",
    ) -> List[List[float]]:
        """
        Convenience method to create embeddings for multiple texts.

        Args:
            texts: List of text strings to embed.
            model: The embedding model to use.
            dimensions: Number of dimensions for the output embeddings.
            encoding_format: Format for the embeddings.

        Returns:
            List of embedding vectors in the same order as input texts.
        """
        response = self.create(
            input=texts,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )

        # Sort by index to ensure correct order
        sorted_data = sorted(response.data, key=lambda x: x.index)

        # Extract embeddings
        if encoding_format == "float":
            return [item.embedding for item in sorted_data]
        else:
            # For base64, return as-is
            return [item.embedding for item in sorted_data]
