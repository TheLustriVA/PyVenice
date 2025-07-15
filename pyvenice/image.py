"""
Image generation, styles, and upscaling endpoints for Venice.ai API.
"""

import base64
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Literal, BinaryIO
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator

from .client import BaseResource, VeniceClient
from .models import Models


class GenerateImageRequest(BaseModel):
    """Request model for Venice image generation."""

    model: str
    prompt: str = Field(..., min_length=1, max_length=1500)

    # Optional parameters
    cfg_scale: Optional[float] = Field(
        None, gt=0, le=20, description="CFG scale parameter"
    )
    embed_exif_metadata: bool = False
    format: Literal["jpeg", "png", "webp"] = "webp"
    height: int = Field(1024, gt=0, le=1280)
    hide_watermark: bool = True
    lora_strength: Optional[int] = Field(None, ge=0, le=100)
    negative_prompt: Optional[str] = Field(None, max_length=1500)
    return_binary: bool = False
    safe_mode: bool = False
    seed: Optional[int] = Field(None, ge=-999999999, le=999999999)
    steps: int = Field(20, gt=0, le=30)
    style_preset: Optional[str] = None
    width: int = Field(1024, gt=0, le=1280)


class OpenAIImageRequest(BaseModel):
    """Request model for OpenAI-compatible image generation."""

    prompt: str = Field(..., min_length=1, max_length=1500)

    # Optional parameters
    model: str = "default"
    background: Optional[Literal["transparent", "opaque", "auto"]] = "auto"
    moderation: Literal["low", "auto"] = "auto"
    n: int = Field(1, ge=1, le=1)  # Venice only supports 1
    output_compression: Optional[int] = Field(None, ge=0, le=100)
    output_format: Literal["jpeg", "png", "webp"] = "png"
    quality: Optional[Literal["auto", "high", "medium", "low", "hd", "standard"]] = (
        "auto"
    )
    response_format: Literal["b64_json", "url"] = "b64_json"
    size: Optional[str] = "auto"
    style: Optional[Literal["vivid", "natural"]] = "natural"
    user: Optional[str] = None

    @field_validator("size")
    @classmethod
    def validate_size(cls, v):
        valid_sizes = {
            "auto",
            "256x256",
            "512x512",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "1792x1024",
            "1024x1792",
        }
        if v and v not in valid_sizes:
            raise ValueError(f"Invalid size. Must be one of: {valid_sizes}")
        return v


class UpscaleImageRequest(BaseModel):
    """Request model for image upscaling."""

    image: Union[str, bytes]  # Base64 string or binary data

    # Optional parameters
    enhance: Union[bool, Literal["true", "false"]] = False
    enhanceCreativity: float = Field(0.5, ge=0, le=1)
    enhancePrompt: Optional[str] = Field(None, max_length=1500)
    replication: float = Field(0.35, ge=0.1, le=1)
    scale: float = Field(2, ge=1, le=4)

    @model_validator(mode="after")
    def validate_enhance_with_scale(self):
        """Validate that enhance must be true when scale is 1."""
        if self.scale == 1 and not self.enhance:
            raise ValueError("enhance must be true when scale is 1")
        return self


class EditImageRequest(BaseModel):
    """Request model for image editing."""
    
    prompt: str = Field(..., min_length=1, max_length=1500, description="Text directions to edit or modify the image")
    image: Union[str, bytes]  # Base64 string, binary data, or URL


class ImageGenerationResponse(BaseModel):
    """Response from image generation endpoints."""

    id: str
    images: List[str]  # Base64 encoded images
    request: Optional[Dict[str, Any]] = None
    timing: Optional[Dict[str, float]] = None
    data: Optional[Any] = Field(None, description="data")
    embed_exif_metadata: Optional[Any] = Field(None, description="embed_exif_metadata")
    format: Optional[Any] = Field(None, description="format")
    height: Optional[Any] = Field(None, description="height")
    hide_watermark: Optional[str] = Field(None, description="hide_watermark")
    model: Optional[Any] = Field(None, description="model")
    prompt: Optional[Any] = Field(None, description="prompt")
    return_binary: Optional[Any] = Field(None, description="return_binary")
    safe_mode: Optional[Any] = Field(None, description="safe_mode")
    seed: Optional[Any] = Field(None, description="seed")


class OpenAIImageResponse(BaseModel):
    """OpenAI-compatible image generation response."""

    created: int
    data: List[Dict[str, str]]  # List of {"b64_json": "..." } or {"url": "..."}


class ImageGeneration(BaseResource):
    """
    Interface for Venice.ai image generation endpoints.

    Provides methods for:
    - Image generation (Venice and OpenAI compatible)
    - Style presets listing
    - Image upscaling and enhancement
    - Image editing based on text prompts
    """

    def __init__(self, client: VeniceClient):
        super().__init__(client)
        self.models = Models(client)
        self._styles_cache: Optional[List[str]] = None

    def generate(
        self,
        prompt: str,
        model: str = "venice-sd35",
        *,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: Optional[int] = None,
        style_preset: Optional[str] = None,
        format: Literal["jpeg", "png", "webp"] = "webp",
        safe_mode: bool = False,
        return_binary: bool = False,
        **kwargs,
    ) -> ImageGenerationResponse:
        """
        Generate an image using Venice's native API.

        Args:
            prompt: Text description of the desired image.
            model: Model to use for generation.
            negative_prompt: What should not be in the image.
            width: Image width (max 1280).
            height: Image height (max 1280).
            steps: Number of inference steps (max 30).
            cfg_scale: CFG scale parameter (0-20).
            seed: Random seed for reproducibility.
            style_preset: Style to apply (see list_styles()).
            format: Output format.
            safe_mode: Blur adult content if detected.
            return_binary: Return binary data instead of base64.
            **kwargs: Additional parameters.

        Returns:
            ImageGenerationResponse with generated images.
        """
        # Map model name if needed
        model = self.models.map_model_name(model)

        request = GenerateImageRequest(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            style_preset=style_preset,
            format=format,
            safe_mode=safe_mode,
            return_binary=return_binary,
            **kwargs,
        )

        response = self.client.post(
            "/image/generate", request.model_dump(exclude_none=True)
        )

        # Check response headers for content violations
        if hasattr(response, "headers"):
            if response.headers.get("x-venice-is-content-violation") == "true":
                print("Warning: Generated image violates content policy")
            if response.headers.get("x-venice-is-blurred") == "true":
                print("Warning: Generated image has been blurred due to adult content")

        return ImageGenerationResponse(**response)

    def generate_openai_style(
        self,
        prompt: str,
        *,
        model: str = "default",
        size: str = "1024x1024",
        quality: str = "auto",
        n: int = 1,
        response_format: Literal["b64_json", "url"] = "b64_json",
        **kwargs,
    ) -> OpenAIImageResponse:
        """
        Generate an image using OpenAI-compatible API.

        Args:
            prompt: Text description of the desired image.
            model: Model to use (defaults to Venice's default).
            size: Image size (e.g., "1024x1024").
            quality: Quality setting (ignored by Venice).
            n: Number of images (Venice only supports 1).
            response_format: Response format.
            **kwargs: Additional parameters.

        Returns:
            OpenAIImageResponse compatible with OpenAI clients.
        """
        request = OpenAIImageRequest(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            n=n,
            response_format=response_format,
            **kwargs,
        )

        response = self.client.post(
            "/images/generations", request.model_dump(exclude_none=True)
        )
        return OpenAIImageResponse(**response)

    async def generate_async(
        self, prompt: str, model: str = "venice-sd35", **kwargs
    ) -> ImageGenerationResponse:
        """Async version of generate()."""
        model = self.models.map_model_name(model)

        request = GenerateImageRequest(model=model, prompt=prompt, **kwargs)

        response = await self.client.post_async(
            "/image/generate", request.model_dump(exclude_none=True)
        )
        return ImageGenerationResponse(**response)

    def list_styles(self, force_refresh: bool = False) -> List[str]:
        """
        List available image style presets.

        Args:
            force_refresh: Force refresh of cached styles.

        Returns:
            List of available style names.
        """
        if not force_refresh and self._styles_cache is not None:
            return self._styles_cache

        response = self.client.get("/image/styles")
        self._styles_cache = response.get("data", [])
        return self._styles_cache

    def upscale(
        self,
        image: Union[str, bytes, Path, BinaryIO],
        *,
        scale: float = 2,
        enhance: bool = False,
        enhance_creativity: float = 0.5,
        enhance_prompt: Optional[str] = None,
        replication: float = 0.35,
    ) -> bytes:
        """
        Upscale or enhance an image.

        Args:
            image: Image to upscale (base64, bytes, file path, or file object).
            scale: Scale factor (1-4). Use 1 with enhance=True for enhancement only.
            enhance: Apply Venice's image enhancement.
            enhance_creativity: How much AI can change the image (0-1).
            enhance_prompt: Style to apply during enhancement.
            replication: Preserve lines/noise from original (0.1-1).

        Returns:
            Upscaled image as bytes.
        """
        # Convert image to base64 if needed
        if isinstance(image, Path):
            with open(image, "rb") as f:
                image = base64.b64encode(f.read()).decode()
        elif isinstance(image, BinaryIO):
            image = base64.b64encode(image.read()).decode()
        elif isinstance(image, bytes):
            image = base64.b64encode(image).decode()

        request = UpscaleImageRequest(
            image=image,
            scale=scale,
            enhance=enhance,
            enhanceCreativity=enhance_creativity,
            enhancePrompt=enhance_prompt,
            replication=replication,
        )

        # This endpoint returns binary data
        headers = {"Accept": "image/png"}
        response = self.client._request(
            "POST",
            "/image/upscale",
            data=request.model_dump(exclude_none=True),
            headers=headers,
            stream=True,
        )

        return response.content

    async def upscale_async(
        self, image: Union[str, bytes, Path, BinaryIO], **kwargs
    ) -> bytes:
        """Async version of upscale()."""
        # Convert image to base64 if needed
        if isinstance(image, Path):
            with open(image, "rb") as f:
                image = base64.b64encode(f.read()).decode()
        elif isinstance(image, BinaryIO):
            image = base64.b64encode(image.read()).decode()
        elif isinstance(image, bytes):
            image = base64.b64encode(image).decode()

        request = UpscaleImageRequest(image=image, **kwargs)

        headers = {"Accept": "image/png"}
        response = await self.client._request_async(
            "POST",
            "/image/upscale",
            data=request.model_dump(exclude_none=True),
            headers=headers,
            stream=True,
        )

        return response.content

    def edit(
        self,
        prompt: str,
        image: Union[str, bytes, Path, BinaryIO],
    ) -> bytes:
        """
        Edit or modify an image based on a text prompt.
        
        Args:
            prompt: Text description of desired edits (max 1500 chars).
            image: Image to edit (base64, bytes, file path, or file object).
            
        Returns:
            Edited image as bytes.
            
        Example:
            >>> # Edit from file
            >>> edited_image = image_gen.edit(
            ...     "Change the sky to a sunset",
            ...     "input.jpg"
            ... )
            >>> with open("edited.png", "wb") as f:
            ...     f.write(edited_image)
        """
        # Convert image to base64 if needed
        if isinstance(image, Path):
            with open(image, "rb") as f:
                image = base64.b64encode(f.read()).decode()
        elif isinstance(image, BinaryIO):
            image = base64.b64encode(image.read()).decode()
        elif isinstance(image, bytes):
            image = base64.b64encode(image).decode()
        elif isinstance(image, str) and not image.startswith(("http://", "https://", "data:")):
            # Assume it's a file path if it's a string but not a URL or base64 data URI
            try:
                with open(image, "rb") as f:
                    image = base64.b64encode(f.read()).decode()
            except (FileNotFoundError, OSError):
                # If it's not a valid file path, assume it's already base64
                pass

        request = EditImageRequest(
            prompt=prompt,
            image=image,
        )

        # This endpoint returns binary data
        headers = {"Accept": "image/png"}
        response = self.client._request(
            "POST",
            "/image/edit",
            data=request.model_dump(exclude_none=True),
            headers=headers,
            stream=True,
        )

        return response.content

    async def edit_async(
        self,
        prompt: str,
        image: Union[str, bytes, Path, BinaryIO],
    ) -> bytes:
        """Async version of edit()."""
        # Convert image to base64 if needed
        if isinstance(image, Path):
            with open(image, "rb") as f:
                image = base64.b64encode(f.read()).decode()
        elif isinstance(image, BinaryIO):
            image = base64.b64encode(image.read()).decode()
        elif isinstance(image, bytes):
            image = base64.b64encode(image).decode()
        elif isinstance(image, str) and not image.startswith(("http://", "https://", "data:")):
            # Assume it's a file path if it's a string but not a URL or base64 data URI
            try:
                with open(image, "rb") as f:
                    image = base64.b64encode(f.read()).decode()
            except (FileNotFoundError, OSError):
                # If it's not a valid file path, assume it's already base64
                pass

        request = EditImageRequest(
            prompt=prompt,
            image=image,
        )

        headers = {"Accept": "image/png"}
        response = await self.client._request_async(
            "POST",
            "/image/edit",
            data=request.model_dump(exclude_none=True),
            headers=headers,
            stream=True,
        )

        return response.content

    def save_images(
        self,
        response: Union[ImageGenerationResponse, OpenAIImageResponse],
        output_dir: Union[str, Path] = ".",
        filename_template: str = "image_{index}_{timestamp}",
        format: str = "png",
        save_metadata: bool = True,
    ) -> List[Path]:
        """
        Save generated images to disk.

        Args:
            response: Image generation response from generate() or generate_openai_style()
            output_dir: Directory to save images (default: current directory)
            filename_template: Template for filenames. Supports placeholders:
                - {index}: Image index (0, 1, 2, ...)
                - {timestamp}: Current timestamp (YYYYMMDD_HHMMSS)
                - {id}: Response ID (for ImageGenerationResponse only)
            format: Image file format extension (png, jpg, jpeg, webp)
            save_metadata: Whether to save generation metadata as JSON sidecar file

        Returns:
            List of Path objects pointing to saved image files

        Example:
            >>> response = image_gen.generate("A sunset", model="venice-sd35")
            >>> saved_paths = image_gen.save_images(
            ...     response,
            ...     output_dir="./outputs",
            ...     filename_template="sunset_{index}",
            ...     format="png"
            ... )
            >>> print(f"Saved {len(saved_paths)} images")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize format
        format = format.lower().lstrip(".")
        if format == "jpg":
            format = "jpeg"

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract images and metadata based on response type
        if isinstance(response, ImageGenerationResponse):
            images = response.images
            response_id = response.id
            metadata = {
                "id": response.id,
                "request": response.request,
                "timing": response.timing,
                "saved_at": datetime.now().isoformat(),
                "format": format,
            }
        elif isinstance(response, OpenAIImageResponse):
            # Extract base64 data from OpenAI format
            images = []
            for item in response.data:
                if "b64_json" in item:
                    images.append(item["b64_json"])
                else:
                    raise ValueError(
                        "OpenAI response contains URLs, not base64 data. Cannot save to disk."
                    )
            response_id = str(response.created)
            metadata = {
                "created": response.created,
                "saved_at": datetime.now().isoformat(),
                "format": format,
            }
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")

        saved_paths = []

        # Save each image
        for index, image_b64 in enumerate(images):
            # Format filename
            filename = filename_template.format(
                index=index,
                timestamp=timestamp,
                id=response_id,
            )

            # Add extension if not present
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            file_path = output_dir / filename

            # Decode and save image
            try:
                image_data = base64.b64decode(image_b64)
                with open(file_path, "wb") as f:
                    f.write(image_data)
                saved_paths.append(file_path)
            except Exception as e:
                raise ValueError(f"Failed to save image {index}: {e}")

        # Save metadata if requested
        if save_metadata and saved_paths:
            metadata_filename = f"{filename_template.format(index='metadata', timestamp=timestamp, id=response_id)}.json"
            metadata_path = output_dir / metadata_filename

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return saved_paths
