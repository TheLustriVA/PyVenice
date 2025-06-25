"""
Chat completions endpoint wrapper for Venice.ai API.
"""

import json
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Union,
    Generator,
    AsyncGenerator,
    Literal,
    TypedDict,
    overload,
)
from pydantic import BaseModel, Field, field_validator

from .client import BaseResource, VeniceClient
from .models import Models
from .validators import validate_model_capabilities


class TextContent(TypedDict):
    """Text content in a message."""

    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    """Image content in a message."""

    type: Literal["image_url"]
    image_url: Dict[str, str]


MessageContent = Union[str, List[Union[TextContent, ImageContent]]]


class Message(TypedDict, total=False):
    """Base message structure."""

    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent
    name: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    tool_call_id: Optional[str]
    reasoning_content: Optional[str]


class VeniceParameters(BaseModel):
    """Venice-specific parameters for chat completions."""

    character_slug: Optional[str] = Field(
        None, description="Character slug for Venice character"
    )
    strip_thinking_response: bool = Field(
        False, description="Strip <think></think> blocks"
    )
    disable_thinking: bool = Field(
        False, description="Disable thinking on reasoning models"
    )
    enable_web_search: Literal["auto", "off", "on"] = Field(
        "off", description="Enable web search"
    )
    enable_web_citations: bool = Field(
        False, description="Enable web citations with [REF] format"
    )
    include_venice_system_prompt: bool = Field(
        True, description="Include Venice system prompts"
    )


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = Field(
        False, description="Include usage information in stream"
    )


class ResponseFormat(BaseModel):
    """Response format specification."""

    type: Literal["json_object", "json_schema", "text"]
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions."""

    model: str
    messages: List[Dict[str, Any]]

    # Optional parameters
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    logprobs: Optional[bool] = None
    max_completion_tokens: Optional[int] = None
    max_temp: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None  # Deprecated, use max_completion_tokens
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_temp: Optional[float] = Field(None, ge=0.0, le=2.0)
    n: int = Field(1, description="Number of completions to generate")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    repetition_penalty: Optional[float] = Field(None, ge=0.0)
    seed: Optional[int] = Field(None, gt=0)
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    user: Optional[str] = None  # Ignored but supported for OpenAI compatibility

    # Function calling parameters
    parallel_tool_calls: Optional[bool] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Response format
    response_format: Optional[Union[Dict[str, Any], ResponseFormat]] = None

    # Venice-specific parameters
    venice_parameters: Optional[VeniceParameters] = None

    # Reasoning parameters
    reasoning_effort: Optional[str] = None

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v):
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("Maximum 4 stop sequences allowed")
        return v


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    finish_reason: Literal["stop", "length"]
    index: int
    message: Dict[str, Any]
    logprobs: Optional[Dict[str, Any]] = None
    stop_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    venice_parameters: Optional[Dict[str, Any]] = None
    prompt_logprobs: Optional[Any] = None
    kv_transfer_params: Optional[Dict[str, Any]] = None
    completion_tokens: Optional[str] = Field(None, description="completion_tokens")
    prompt_tokens: Optional[str] = Field(None, description="prompt_tokens")
    prompt_tokens_details: Optional[str] = Field(
        None, description="prompt_tokens_details"
    )
    total_tokens: Optional[str] = Field(None, description="total_tokens")
    disable_thinking: Optional[Any] = Field(None, description="disable_thinking")


class ChatCompletionChunk(BaseModel):
    """A single chunk in a streaming response."""

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Usage] = None
    venice_parameters: Optional[Dict[str, Any]] = None


class ChatCompletion(BaseResource):
    """
    Interface for Venice.ai chat completions endpoint.

    Provides methods to create chat completions with support for:
    - Standard and streaming responses
    - Function calling
    - Web search
    - Venice-specific features
    - Model capability validation
    """

    def __init__(self, client: VeniceClient):
        super().__init__(client)
        self.models = Models(client)

    @overload
    def create(
        self,
        *,
        model: str,
        messages: List[Message],
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletionResponse: ...

    @overload
    def create(
        self, *, model: str, messages: List[Message], stream: Literal[True], **kwargs
    ) -> Generator[ChatCompletionChunk, None, None]: ...

    @validate_model_capabilities(auto_remove_unsupported=True)
    def create(
        self, *, model: str, messages: List[Message], stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]:
        """
        Create a chat completion.

        Args:
            model: The model ID to use.
            messages: List of messages in the conversation.
            stream: Whether to stream the response.
            **kwargs: Additional parameters from ChatCompletionRequest.

        Returns:
            ChatCompletionResponse if stream=False, Generator of chunks if stream=True.
        """
        # Map model name if needed
        model = self.models.map_model_name(model)

        # Build request
        request_data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        # Validate request
        request = ChatCompletionRequest(**request_data)

        # Convert to dict, excluding None values
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._create_stream(payload)
        else:
            response = self.client.post("/chat/completions", payload)
            return ChatCompletionResponse(**response)

    def _create_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[ChatCompletionChunk, None, None]:
        """Handle streaming response."""
        response = self.client.post("/chat/completions", payload, stream=True)

        # httpx Response doesn't need context manager when streaming
        for line in response.iter_lines():
            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(data)
                    yield ChatCompletionChunk(**chunk_data)
                except json.JSONDecodeError:
                    continue

    @overload
    async def create_async(
        self,
        *,
        model: str,
        messages: List[Message],
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletionResponse: ...

    @overload
    async def create_async(
        self, *, model: str, messages: List[Message], stream: Literal[True], **kwargs
    ) -> AsyncGenerator[ChatCompletionChunk, None]: ...

    @validate_model_capabilities(auto_remove_unsupported=True)
    async def create_async(
        self, *, model: str, messages: List[Message], stream: bool = False, **kwargs
    ) -> Union[ChatCompletionResponse, AsyncGenerator[ChatCompletionChunk, None]]:
        """
        Async version of create().

        Args:
            model: The model ID to use.
            messages: List of messages in the conversation.
            stream: Whether to stream the response.
            **kwargs: Additional parameters from ChatCompletionRequest.

        Returns:
            ChatCompletionResponse if stream=False, AsyncGenerator of chunks if stream=True.
        """
        # Map model name if needed
        model = self.models.map_model_name(model)

        # Build request
        request_data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs,
        }

        # Validate request
        request = ChatCompletionRequest(**request_data)

        # Convert to dict, excluding None values
        payload = request.model_dump(exclude_none=True)

        if stream:
            return self._create_stream_async(payload)
        else:
            response = await self.client.post_async("/chat/completions", payload)
            return ChatCompletionResponse(**response)

    async def _create_stream_async(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Handle async streaming response."""
        response = await self.client.post_async(
            "/chat/completions", payload, stream=True
        )

        # Use the response directly for async iteration
        async for line in response.aiter_lines():
            if not line:
                continue

            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(data)
                    yield ChatCompletionChunk(**chunk_data)
                except json.JSONDecodeError:
                    continue

    def create_with_web_search(
        self,
        *,
        model: str,
        messages: List[Message],
        enable_citations: bool = True,
        search_mode: Literal["auto", "on"] = "auto",
        **kwargs,
    ) -> Union[ChatCompletionResponse, Generator[ChatCompletionChunk, None, None]]:
        """
        Create a chat completion with web search enabled.

        Args:
            model: The model ID to use.
            messages: List of messages in the conversation.
            enable_citations: Whether to include web citations.
            search_mode: Web search mode ("auto" or "on").
            **kwargs: Additional parameters.

        Returns:
            Chat completion response or stream.
        """
        venice_params = kwargs.get("venice_parameters", {})
        if isinstance(venice_params, VeniceParameters):
            venice_params = venice_params.model_dump()

        venice_params.update(
            {
                "enable_web_search": search_mode,
                "enable_web_citations": enable_citations,
            }
        )

        kwargs["venice_parameters"] = venice_params

        return self.create(model=model, messages=messages, **kwargs)
