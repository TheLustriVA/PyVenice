"""
Models endpoint wrapper for Venice.ai API.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from .client import BaseResource, VeniceClient


class ModelCapabilities(BaseModel):
    """Model capabilities information."""

    optimizedForCode: bool = Field(description="Is the LLM optimized for coding?")
    quantization: Literal["fp8", "fp16", "not-available"] = Field(
        description="The quantization type"
    )
    supportsFunctionCalling: bool = Field(
        description="Does the model support function calling?"
    )
    supportsReasoning: bool = Field(
        description="Does the model support reasoning with <thinking> blocks?"
    )
    supportsResponseSchema: bool = Field(
        description="Does the model support response schema?"
    )
    supportsVision: bool = Field(description="Does the model support vision?")
    supportsWebSearch: bool = Field(description="Does the model support web search?")
    supportsLogProbs: bool = Field(
        description="Does the model support logprobs parameter?"
    )


class ModelSpec(BaseModel):
    """Model specification details."""

    availableContextTokens: Optional[int] = Field(
        None, description="Context length for text models"
    )
    beta: bool = Field(False, description="Is this model in beta?")
    capabilities: Optional[ModelCapabilities] = Field(
        None, description="Model capabilities"
    )
    costPerInToken: Optional[float] = Field(None, description="Cost per input token")
    costPerOutToken: Optional[float] = Field(None, description="Cost per output token")
    dailyLimit: Optional[int] = Field(None, description="Daily usage limit")
    defaultParameters: Optional[Dict[str, Any]] = Field(
        None, description="Default parameters"
    )
    description: Optional[str] = Field(None, description="Model description")
    developer: Optional[str] = Field(None, description="Model developer")
    disabledInPlayground: Optional[bool] = Field(
        None, description="Disabled in playground?"
    )
    earliestAccess: Optional[str] = Field(None, description="Earliest access tier")
    hasDisclaimerPopup: Optional[bool] = Field(
        None, description="Has disclaimer popup?"
    )
    humanName: Optional[str] = Field(None, description="Human-readable name")
    imageGenerations: Optional[int] = Field(
        None, description="Number of image generations"
    )
    isAvailable: Optional[bool] = Field(None, description="Is model available?")
    maxCompletionTokens: Optional[int] = Field(
        None, description="Maximum completion tokens"
    )
    modelType: Optional[str] = Field(None, description="Type of model")
    requestsPerMin: Optional[int] = Field(None, description="Requests per minute limit")
    runpodInstanceId: Optional[str] = Field(None, description="Runpod instance ID")
    tags: Optional[List[str]] = Field(None, description="Model tags")
    upscalingFactor: Optional[int] = Field(
        None, description="Upscaling factor for image models"
    )


class Model(BaseModel):
    """Model information."""

    created: int = Field(description="Release date on Venice API")
    id: str = Field(description="Model ID")
    model_spec: ModelSpec = Field(description="Model specifications")
    object: Literal["model"] = Field("model", description="Object type")
    owned_by: str = Field(description="Model owner")
    type: Optional[str] = Field(None, description="Model type")


class ModelListResponse(BaseModel):
    """Response from models list endpoint."""

    data: List[Model] = Field(description="List of available models")
    object: Literal["list"] = Field("list", description="Response object type")
    type: Optional[str] = Field(None, description="Filter type used")


class ModelCompatibilityMapping(BaseModel):
    """Model compatibility mapping information."""

    original_model: str = Field(description="Original model name")
    mapped_model: str = Field(description="Mapped Venice model")
    type: str = Field(description="Model type")


class ModelTraits(BaseModel):
    """Model traits information."""

    model_types: List[str] = Field(description="Available model types")
    model_tags: List[str] = Field(description="Available model tags")


class Models(BaseResource):
    """
    Interface for Venice.ai models endpoints.

    Provides methods to list models, get model capabilities, and check compatibility.
    """

    # Cache duration for model capabilities (24 hours)
    CACHE_DURATION = timedelta(hours=24)

    def __init__(self, client: VeniceClient):
        super().__init__(client)
        self._models_cache: Optional[ModelListResponse] = None
        self._models_cache_time: Optional[datetime] = None
        self._compatibility_cache: Optional[Dict[str, str]] = None
        self._traits_cache: Optional[ModelTraits] = None

    def list(
        self,
        type: Optional[
            Literal["embedding", "image", "text", "tts", "upscale", "all", "code"]
        ] = None,
        force_refresh: bool = False,
    ) -> ModelListResponse:
        """
        List available models.

        Args:
            type: Filter models by type. Use "all" to get all model types.
            force_refresh: Force refresh of cached data.

        Returns:
            ModelListResponse with list of available models.
        """
        # Check cache if no type filter and not forcing refresh
        if not type and not force_refresh and self._models_cache:
            if (
                self._models_cache_time
                and datetime.now() - self._models_cache_time < self.CACHE_DURATION
            ):
                return self._models_cache

        params = {}
        if type:
            params["type"] = type

        response = self.client.get("/models", params=params)
        result = ModelListResponse(**response)

        # Cache the result if it's an unfiltered request
        if not type:
            self._models_cache = result
            self._models_cache_time = datetime.now()

        return result

    async def list_async(
        self,
        type: Optional[
            Literal["embedding", "image", "text", "tts", "upscale", "all", "code"]
        ] = None,
        force_refresh: bool = False,
    ) -> ModelListResponse:
        """Async version of list()."""
        # Check cache if no type filter and not forcing refresh
        if not type and not force_refresh and self._models_cache:
            if (
                self._models_cache_time
                and datetime.now() - self._models_cache_time < self.CACHE_DURATION
            ):
                return self._models_cache

        params = {}
        if type:
            params["type"] = type

        response = await self.client.get_async("/models", params=params)
        result = ModelListResponse(**response)

        # Cache the result if it's an unfiltered request
        if not type:
            self._models_cache = result
            self._models_cache_time = datetime.now()

        return result

    def get_model(self, model_id: str, force_refresh: bool = False) -> Optional[Model]:
        """
        Get a specific model by ID.

        Args:
            model_id: The model ID to retrieve.
            force_refresh: Force refresh of cached data.

        Returns:
            Model object if found, None otherwise.
        """
        models = self.list(force_refresh=force_refresh)
        for model in models.data:
            if model.id == model_id:
                return model
        return None

    def get_capabilities(
        self, model_id: str, force_refresh: bool = False
    ) -> Optional[ModelCapabilities]:
        """
        Get capabilities for a specific model.

        Args:
            model_id: The model ID to get capabilities for.
            force_refresh: Force refresh of cached data.

        Returns:
            ModelCapabilities if model found and has capabilities, None otherwise.
        """
        model = self.get_model(model_id, force_refresh=force_refresh)
        if model and model.model_spec.capabilities:
            return model.model_spec.capabilities
        return None

    def supports_parameter(
        self, model_id: str, parameter: str, force_refresh: bool = False
    ) -> bool:
        """
        Check if a model supports a specific parameter.

        Args:
            model_id: The model ID to check.
            parameter: The parameter name to check (e.g., 'parallel_tool_calls', 'logprobs').
            force_refresh: Force refresh of cached data.

        Returns:
            True if the model supports the parameter, False otherwise.
        """
        capabilities = self.get_capabilities(model_id, force_refresh=force_refresh)
        if not capabilities:
            return False

        # Map common parameters to capability fields
        parameter_mapping = {
            "parallel_tool_calls": "supportsFunctionCalling",
            "tools": "supportsFunctionCalling",
            "tool_choice": "supportsFunctionCalling",
            "functions": "supportsFunctionCalling",
            "function_call": "supportsFunctionCalling",
            "response_format": "supportsResponseSchema",
            "response_schema": "supportsResponseSchema",
            "logprobs": "supportsLogProbs",
            "top_logprobs": "supportsLogProbs",
            "reasoning_effort": "supportsReasoning",
        }

        capability_field = parameter_mapping.get(parameter)
        if capability_field:
            return getattr(capabilities, capability_field, False)

        return True  # Default to True for unknown parameters

    def get_traits(self, force_refresh: bool = False) -> ModelTraits:
        """
        Get model traits (types and tags).

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            ModelTraits with available types and tags.
        """
        if not force_refresh and self._traits_cache:
            return self._traits_cache

        response = self.client.get("/models/traits")
        self._traits_cache = ModelTraits(**response)
        return self._traits_cache

    def get_compatibility_mapping(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get model compatibility mapping.

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            Dictionary mapping original model names to Venice model names.
        """
        if not force_refresh and self._compatibility_cache:
            return self._compatibility_cache

        response = self.client.get("/models/compatibility_mapping")

        # The data is already a dictionary mapping
        self._compatibility_cache = response.get("data", {})

        return self._compatibility_cache

    def map_model_name(self, model_name: str, force_refresh: bool = False) -> str:
        """
        Map an OpenAI or other model name to Venice equivalent.

        Args:
            model_name: The original model name.
            force_refresh: Force refresh of cached data.

        Returns:
            The mapped Venice model name, or original if no mapping exists.
        """
        mapping = self.get_compatibility_mapping(force_refresh=force_refresh)
        return mapping.get(model_name, model_name)
