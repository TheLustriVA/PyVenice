"""Unit tests for the ImageGeneration module."""

import pytest
import httpx
import respx
import base64
from pathlib import Path
from unittest.mock import patch, mock_open

from pyvenice.image import ImageGeneration, GenerateImageRequest, OpenAIImageRequest, EditImageRequest
from pyvenice.client import VeniceClient
from pydantic import ValidationError


@pytest.mark.unit
class TestImageGeneration:
    """Test ImageGeneration class functionality."""

    def test_image_generation_init(self, client):
        """Test ImageGeneration initialization."""
        image_gen = ImageGeneration(client)
        assert image_gen.client == client
        assert hasattr(image_gen, "models")
        assert image_gen._styles_cache is None

    @respx.mock
    def test_generate_image(
        self, respx_mock, mock_image_response, mock_compatibility_mapping, client
    ):
        """Test generating an image."""
        # Mock endpoints
        respx_mock.get(
            "https://api.venice.ai/api/v1/models/compatibility_mapping"
        ).mock(return_value=httpx.Response(200, json=mock_compatibility_mapping))
        respx_mock.post("https://api.venice.ai/api/v1/image/generate").mock(
            return_value=httpx.Response(200, json=mock_image_response)
        )

        image_gen = ImageGeneration(client)
        response = image_gen.generate(prompt="A beautiful sunset", model="venice-sd35")

        assert response.id == "generate-image-test123"
        assert len(response.images) == 1
        assert response.timing["total"] == 2650

    @respx.mock
    def test_generate_image_with_params(
        self, respx_mock, mock_image_response, mock_compatibility_mapping, client
    ):
        """Test generating image with custom parameters."""
        # Mock endpoints
        respx_mock.get(
            "https://api.venice.ai/api/v1/models/compatibility_mapping"
        ).mock(return_value=httpx.Response(200, json=mock_compatibility_mapping))
        respx_mock.post("https://api.venice.ai/api/v1/image/generate").mock(
            return_value=httpx.Response(200, json=mock_image_response)
        )

        image_gen = ImageGeneration(client)
        image_gen.generate(
            prompt="A cyberpunk city",
            model="venice-sd35",
            width=512,
            height=512,
            steps=30,
            cfg_scale=8.0,
            negative_prompt="blurry, low quality",
            style_preset="Cyberpunk",
        )

        # Verify request was made with correct parameters
        request = respx_mock.calls[-1].request
        import json

        request_body = json.loads(request.content)

        assert request_body["prompt"] == "A cyberpunk city"
        assert request_body["width"] == 512
        assert request_body["height"] == 512
        assert request_body["steps"] == 30
        assert request_body["cfg_scale"] == 8.0
        assert request_body["negative_prompt"] == "blurry, low quality"
        assert request_body["style_preset"] == "Cyberpunk"

    @respx.mock
    def test_generate_openai_style(self, respx_mock, client):
        """Test OpenAI-compatible image generation."""
        openai_response = {
            "created": 1742262554,
            "data": [
                {
                    "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA0"
                }
            ],
        }

        respx_mock.post("https://api.venice.ai/api/v1/images/generations").mock(
            return_value=httpx.Response(200, json=openai_response)
        )

        image_gen = ImageGeneration(client)
        response = image_gen.generate_openai_style(
            prompt="A forest scene", size="1024x1024", quality="high"
        )

        assert response.created == 1742262554
        assert len(response.data) == 1
        assert "b64_json" in response.data[0]

    @respx.mock
    def test_list_styles(self, respx_mock, client):
        """Test listing image styles."""
        styles_response = {
            "data": ["3D Model", "Anime", "Cinematic", "Comic Book"],
            "object": "list",
        }

        respx_mock.get("https://api.venice.ai/api/v1/image/styles").mock(
            return_value=httpx.Response(200, json=styles_response)
        )

        image_gen = ImageGeneration(client)
        styles = image_gen.list_styles()

        assert styles == ["3D Model", "Anime", "Cinematic", "Comic Book"]

    @respx.mock
    def test_list_styles_caching(self, respx_mock, client):
        """Test that styles are cached."""
        styles_response = {"data": ["Style1", "Style2"], "object": "list"}

        respx_mock.get("https://api.venice.ai/api/v1/image/styles").mock(
            return_value=httpx.Response(200, json=styles_response)
        )

        image_gen = ImageGeneration(client)

        # First call
        styles1 = image_gen.list_styles()
        assert len(respx_mock.calls) == 1

        # Second call should use cache
        styles2 = image_gen.list_styles()
        assert len(respx_mock.calls) == 1
        assert styles1 == styles2

    @respx.mock
    def test_upscale_image_base64(self, respx_mock, client):
        """Test upscaling image from base64."""
        # Mock binary response
        fake_image_data = b"fake-upscaled-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/upscale").mock(
            return_value=httpx.Response(
                200, content=fake_image_data, headers={"content-type": "image/png"}
            )
        )

        image_gen = ImageGeneration(client)
        input_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA0"
        )

        result = image_gen.upscale(image=input_image, scale=2, enhance=True)

        assert result == fake_image_data

    @respx.mock
    def test_upscale_image_from_bytes(self, respx_mock, client):
        """Test upscaling image from bytes."""
        fake_image_data = b"fake-upscaled-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/upscale").mock(
            return_value=httpx.Response(
                200, content=fake_image_data, headers={"content-type": "image/png"}
            )
        )

        image_gen = ImageGeneration(client)
        input_bytes = b"fake-input-image-data"

        result = image_gen.upscale(image=input_bytes, scale=4, enhance=False)

        assert result == fake_image_data

    @patch("builtins.open", new_callable=mock_open, read_data=b"fake-file-data")
    @respx.mock
    def test_upscale_image_from_file(self, mock_file, respx_mock, client):
        """Test upscaling image from file path."""
        fake_image_data = b"fake-upscaled-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/upscale").mock(
            return_value=httpx.Response(
                200, content=fake_image_data, headers={"content-type": "image/png"}
            )
        )

        image_gen = ImageGeneration(client)

        result = image_gen.upscale(image=Path("/fake/path/image.jpg"), scale=2)

        assert result == fake_image_data
        mock_file.assert_called_once_with(Path("/fake/path/image.jpg"), "rb")

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_async(
        self, respx_mock, mock_image_response, mock_compatibility_mapping, client
    ):
        """Test async image generation."""
        # Mock endpoints
        respx_mock.get(
            "https://api.venice.ai/api/v1/models/compatibility_mapping"
        ).mock(return_value=httpx.Response(200, json=mock_compatibility_mapping))
        respx_mock.post("https://api.venice.ai/api/v1/image/generate").mock(
            return_value=httpx.Response(200, json=mock_image_response)
        )

        image_gen = ImageGeneration(client)
        response = await image_gen.generate_async(
            prompt="An async sunset", model="venice-sd35"
        )

        assert response.id == "generate-image-test123"
        assert len(response.images) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_upscale_async(self, respx_mock, client):
        """Test async image upscaling."""
        fake_image_data = b"fake-upscaled-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/upscale").mock(
            return_value=httpx.Response(
                200, content=fake_image_data, headers={"content-type": "image/png"}
            )
        )

        image_gen = ImageGeneration(client)
        input_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA0"
        )

        result = await image_gen.upscale_async(image=input_image, scale=2)

        assert result == fake_image_data


@pytest.mark.unit
class TestImageRequestValidation:
    """Test image request validation."""

    def test_generate_image_request_valid(self):
        """Test valid image generation request."""
        request = GenerateImageRequest(
            model="venice-sd35", prompt="A beautiful landscape"
        )

        assert request.model == "venice-sd35"
        assert request.prompt == "A beautiful landscape"
        assert request.width == 1024  # Default
        assert request.height == 1024  # Default
        assert request.steps == 20  # Default
        assert request.format == "webp"  # Default

    def test_generate_image_request_with_params(self):
        """Test image request with custom parameters."""
        request = GenerateImageRequest(
            model="test-model",
            prompt="Test prompt",
            width=512,
            height=768,
            steps=25,
            cfg_scale=7.5,
            format="png",
            safe_mode=False,
        )

        assert request.width == 512
        assert request.height == 768
        assert request.steps == 25
        assert request.cfg_scale == 7.5
        assert request.format == "png"
        assert request.safe_mode is False

    def test_generate_image_request_validation_errors(self):
        """Test image request validation errors."""
        from pydantic import ValidationError

        # Empty prompt
        with pytest.raises(ValidationError):
            GenerateImageRequest(model="test", prompt="")

        # Invalid dimensions
        with pytest.raises(ValidationError):
            GenerateImageRequest(model="test", prompt="test", width=0)

        # Invalid CFG scale
        with pytest.raises(ValidationError):
            GenerateImageRequest(
                model="test", prompt="test", cfg_scale=25.0  # Max is 20
            )

    def test_openai_image_request_valid(self):
        """Test valid OpenAI image request."""
        request = OpenAIImageRequest(prompt="A test image")

        assert request.prompt == "A test image"
        assert request.model == "default"
        assert request.size == "auto"
        assert request.response_format == "b64_json"

    def test_openai_image_request_validation(self):
        """Test OpenAI image request validation."""
        from pydantic import ValidationError

        # Invalid size
        with pytest.raises(ValidationError):
            OpenAIImageRequest(prompt="test", size="invalid-size")

        # Too many images (Venice only supports 1)
        with pytest.raises(ValidationError):
            OpenAIImageRequest(prompt="test", n=2)


@pytest.mark.integration
class TestImageIntegration:
    """Integration tests for ImageGeneration (requires API key)."""

    def test_real_image_generation(self, skip_if_no_api_key, integration_api_key):
        """Test real image generation with API."""
        client = VeniceClient(api_key=integration_api_key)
        image_gen = ImageGeneration(client)

        response = image_gen.generate(
            prompt="A simple test image",
            model="venice-sd35",
            width=512,
            height=512,
            steps=10,  # Minimal steps for faster test
        )

        assert response.id
        assert len(response.images) == 1
        assert response.images[0]  # Should have base64 image data

        # Verify it's valid base64
        try:
            base64.b64decode(response.images[0])
        except Exception:
            pytest.fail("Response image is not valid base64")

    def test_real_styles_list(self, skip_if_no_api_key, integration_api_key):
        """Test listing real image styles."""
        client = VeniceClient(api_key=integration_api_key)
        image_gen = ImageGeneration(client)

        styles = image_gen.list_styles()

        assert isinstance(styles, list)
        assert len(styles) > 0
        assert all(isinstance(style, str) for style in styles)


@pytest.mark.unit
class TestImageEdit:
    """Test image editing functionality."""
    
    @respx.mock
    def test_edit_image_basic(self, respx_mock, client):
        """Test basic image editing."""
        # Mock binary response
        fake_edited_image = b"fake-edited-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/edit").mock(
            return_value=httpx.Response(
                200, content=fake_edited_image, headers={"content-type": "image/png"}
            )
        )
        
        image_gen = ImageGeneration(client)
        
        # Test with base64 string
        result = image_gen.edit(
            prompt="Make the sky blue",
            image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA0"
        )
        
        assert result == fake_edited_image
    
    @respx.mock
    def test_edit_image_with_bytes(self, respx_mock, client):
        """Test image editing with bytes input."""
        fake_edited_image = b"fake-edited-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/edit").mock(
            return_value=httpx.Response(
                200, content=fake_edited_image, headers={"content-type": "image/png"}
            )
        )
        
        image_gen = ImageGeneration(client)
        input_bytes = b"fake-input-image-data"
        
        result = image_gen.edit(
            prompt="Change the lighting",
            image=input_bytes
        )
        
        assert result == fake_edited_image
    
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake-file-data")
    @respx.mock
    def test_edit_image_from_file(self, mock_file, respx_mock, client):
        """Test image editing with file input."""
        fake_edited_image = b"fake-edited-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/edit").mock(
            return_value=httpx.Response(
                200, content=fake_edited_image, headers={"content-type": "image/png"}
            )
        )
        
        image_gen = ImageGeneration(client)
        
        result = image_gen.edit(
            prompt="Add a rainbow",
            image=Path("/fake/path/input.jpg")
        )
        
        assert result == fake_edited_image
        mock_file.assert_called_once_with(Path("/fake/path/input.jpg"), "rb")
    
    @respx.mock
    def test_edit_image_file_path_as_string(self, respx_mock, client):
        """Test image editing with file path as string."""
        fake_edited_image = b"fake-edited-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/edit").mock(
            return_value=httpx.Response(
                200, content=fake_edited_image, headers={"content-type": "image/png"}
            )
        )
        
        image_gen = ImageGeneration(client)
        
        with patch("builtins.open", mock_open(read_data=b"fake-file-data")):
            result = image_gen.edit(
                prompt="Change colors",
                image="/fake/path/input.jpg"
            )
        
        assert result == fake_edited_image
    
    @respx.mock
    def test_edit_image_url_handling(self, respx_mock, client):
        """Test that URLs are passed through without file processing."""
        fake_edited_image = b"fake-edited-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/edit").mock(
            return_value=httpx.Response(
                200, content=fake_edited_image, headers={"content-type": "image/png"}
            )
        )
        
        image_gen = ImageGeneration(client)
        
        result = image_gen.edit(
            prompt="Enhance image",
            image="https://example.com/image.jpg"
        )
        
        assert result == fake_edited_image
        
        # Verify the request was made with the URL directly
        request = respx_mock.calls[-1].request
        import json
        request_body = json.loads(request.content)
        assert request_body["image"] == "https://example.com/image.jpg"

    @pytest.mark.asyncio
    @respx.mock
    async def test_edit_image_async(self, respx_mock, client):
        """Test async image editing."""
        fake_edited_image = b"fake-edited-image-data"
        respx_mock.post("https://api.venice.ai/api/v1/image/edit").mock(
            return_value=httpx.Response(
                200, content=fake_edited_image, headers={"content-type": "image/png"}
            )
        )
        
        image_gen = ImageGeneration(client)
        
        result = await image_gen.edit_async(
            prompt="Make it artistic",
            image="base64_encoded_image_data"
        )
        
        assert result == fake_edited_image


@pytest.mark.unit
class TestEditImageRequestValidation:
    """Test EditImageRequest validation."""
    
    def test_edit_image_request_valid(self):
        """Test valid edit image request."""
        request = EditImageRequest(
            prompt="Change the sky color",
            image="base64_encoded_image_data"
        )
        
        assert request.prompt == "Change the sky color"
        assert request.image == "base64_encoded_image_data"
    
    def test_edit_image_request_empty_prompt(self):
        """Test edit image request with empty prompt."""
        with pytest.raises(ValidationError) as exc_info:
            EditImageRequest(
                prompt="",
                image="base64_data"
            )
        
        assert "at least 1 character" in str(exc_info.value)
    
    def test_edit_image_request_long_prompt(self):
        """Test edit image request with too long prompt."""
        long_prompt = "x" * 1501  # Exceeds 1500 character limit
        
        with pytest.raises(ValidationError) as exc_info:
            EditImageRequest(
                prompt=long_prompt,
                image="base64_data"
            )
        
        assert "at most 1500 characters" in str(exc_info.value)
    
    def test_edit_image_request_missing_fields(self):
        """Test edit image request with missing required fields."""
        with pytest.raises(ValidationError):
            EditImageRequest(prompt="test")  # Missing image
        
        with pytest.raises(ValidationError):
            EditImageRequest(image="base64_data")  # Missing prompt


@pytest.mark.integration
class TestImageEditIntegration:
    """Integration tests for image editing (requires API key)."""
    
    def test_real_image_edit(self, skip_if_no_api_key, integration_api_key):
        """Test real image editing with Venice.ai API."""
        client = VeniceClient(api_key=integration_api_key)
        image_gen = ImageGeneration(client)
        
        # Use a simple base64 encoded 1x1 pixel PNG image for testing
        tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        result = image_gen.edit(
            prompt="Make this image blue",
            image=tiny_image
        )
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # Verify it's PNG data (starts with PNG header)
        assert result.startswith(b'\x89PNG\r\n\x1a\n')
    
    @pytest.mark.asyncio
    async def test_real_image_edit_async(self, skip_if_no_api_key, integration_api_key):
        """Test real async image editing with Venice.ai API."""
        client = VeniceClient(api_key=integration_api_key)
        image_gen = ImageGeneration(client)
        
        # Use a simple base64 encoded 1x1 pixel PNG image for testing
        tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        result = await image_gen.edit_async(
            prompt="Add bright colors",
            image=tiny_image
        )
        
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # Verify it's PNG data (starts with PNG header)
        assert result.startswith(b'\x89PNG\r\n\x1a\n')
