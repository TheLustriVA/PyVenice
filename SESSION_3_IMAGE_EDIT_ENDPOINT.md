# Session 3: Image Edit Endpoint Implementation

## Context from Previous Sessions

**Session 1 Results**: Foundation dependencies resolved, API specification updated to v20250713.224148
**Session 2 Results**: Schema fixes completed, integration tests passing

## Prerequisites

Before starting this session, verify previous sessions were completed successfully:
```bash
# Verify you're in the correct directory
cd /home/websinthe/code/pyvenice

# Verify virtual environment and dependencies
source .venv/bin/activate
python -c "import bs4; print('✅ bs4 available')"
which ruff && echo "✅ ruff available"

# Verify API version is current
grep "version:" docs/swagger.yaml | grep "20250713.224148" && echo "✅ API version current"

# Verify integration tests are passing
pytest tests/test_billing.py::TestBillingIntegration::test_real_usage_retrieval -v | grep PASSED && echo "✅ Billing tests fixed"
pytest tests/test_embeddings.py::TestEmbeddingsIntegration::test_real_embeddings_with_dimensions -v | grep PASSED && echo "✅ Embeddings tests fixed"
```

## Goals

1. **Implement New Endpoint**: Add support for POST `/image/edit` endpoint
2. **Create Request Model**: Implement EditImageRequest with proper validation
3. **Add Image Class Method**: Add edit() method to Image class with sync/async support
4. **Comprehensive Testing**: Write unit and integration tests
5. **Documentation**: Update docstrings and examples

## Context from API Audit

### New Endpoint Discovered:
- **Endpoint**: `POST /image/edit`
- **Operation ID**: `editImage`
- **Description**: Edit or modify an image based on the supplied prompt
- **Status**: Completely missing from PyVenice implementation

### API Specification from WebFetch:
The endpoint was detected during API monitoring but is not in the local swagger.yaml (v20250612.151220). It's present in the live API (v20250713.224148).

## Detailed Tasks

### Task 1: Analyze API Specification for Image Edit

```bash
# Search for EditImageRequest in updated swagger.yaml
grep -n -A 20 "EditImageRequest" docs/swagger.yaml

# Search for /image/edit endpoint
grep -n -A 30 "/image/edit" docs/swagger.yaml

# If not found in local swagger, fetch from live API
python -c "
import requests
import yaml

# Fetch live API spec
response = requests.get('https://api.venice.ai/doc/api/swagger.yaml')
spec = yaml.safe_load(response.text)

# Look for image/edit endpoint
paths = spec.get('paths', {})
if '/image/edit' in paths:
    print('Found /image/edit endpoint:')
    print(yaml.dump(paths['/image/edit'], default_flow_style=False))
    
# Look for EditImageRequest schema
schemas = spec.get('components', {}).get('schemas', {})
if 'EditImageRequest' in schemas:
    print('\\nFound EditImageRequest schema:')
    print(yaml.dump(schemas['EditImageRequest'], default_flow_style=False))
"
```

### Task 2: Study Existing Image Module Pattern

**File**: `pyvenice/image.py`

```bash
# Examine existing image.py structure
head -100 pyvenice/image.py

# Look at existing request models
grep -n "class.*Request" pyvenice/image.py

# Look at existing Image class methods
grep -n "def " pyvenice/image.py | grep -E "(generate|upscale|list)"

# Study the generate method pattern
grep -n -A 20 "def generate" pyvenice/image.py
```

**Key Patterns to Follow**:
1. Request model with Pydantic validation
2. Sync and async method variants
3. Error handling and response parsing
4. Parameter validation with @validate_model_params decorator

### Task 3: Implement EditImageRequest Model

**File**: `pyvenice/image.py`

**Add after existing request models**:

```python
class EditImageRequest(BaseModel):
    """Request model for image editing."""
    
    prompt: str = Field(..., min_length=1, max_length=1500)
    image: Union[str, bytes]  # Base64 string or binary data
    
    # Optional parameters (update based on actual API spec)
    model: Optional[str] = None
    mask: Optional[Union[str, bytes]] = None
    strength: Optional[float] = Field(None, ge=0.0, le=1.0)
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0)
    num_inference_steps: Optional[int] = Field(None, ge=1, le=100)
    seed: Optional[int] = Field(None, ge=-999999999, le=999999999)
    
    # Add any other parameters found in API spec
```

**Important**: Update the model based on actual API specification found in Task 1.

### Task 4: Add edit() Method to Image Class

**File**: `pyvenice/image.py`

**Add after existing methods in Image class**:

```python
def edit(
    self,
    prompt: str,
    image: Union[str, bytes, Path],
    *,
    model: Optional[str] = None,
    mask: Optional[Union[str, bytes, Path]] = None,
    strength: Optional[float] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> ImageGenerationResponse:
    """
    Edit an image based on a text prompt.
    
    Args:
        prompt: Text description of desired edits
        image: Input image (base64 string, bytes, or file path)
        model: Model to use for editing (optional)
        mask: Mask to specify edit areas (optional)
        strength: Strength of the edit (0.0 to 1.0)
        guidance_scale: Guidance scale for the edit
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
        **kwargs: Additional parameters
        
    Returns:
        ImageGenerationResponse with edited image
        
    Raises:
        VeniceAPIError: If the API request fails
    """
    
    # Process image input (handle file paths, base64, bytes)
    if isinstance(image, (str, Path)):
        if Path(image).exists():
            # Read from file
            with open(image, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        else:
            # Assume it's already base64
            image_data = image
    elif isinstance(image, bytes):
        image_data = base64.b64encode(image).decode('utf-8')
    else:
        raise ValueError("Image must be a file path, base64 string, or bytes")
    
    # Create request
    request = EditImageRequest(
        prompt=prompt,
        image=image_data,
        model=model,
        mask=mask,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        **kwargs
    )
    
    # Make API call
    response = self.client.post(
        "/image/edit",
        request.model_dump(exclude_none=True)
    )
    
    return ImageGenerationResponse.model_validate(response)

async def edit_async(
    self,
    prompt: str,
    image: Union[str, bytes, Path],
    *,
    model: Optional[str] = None,
    mask: Optional[Union[str, bytes, Path]] = None,
    strength: Optional[float] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> ImageGenerationResponse:
    """Async version of edit()."""
    
    # Same image processing logic as sync version
    if isinstance(image, (str, Path)):
        if Path(image).exists():
            with open(image, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        else:
            image_data = image
    elif isinstance(image, bytes):
        image_data = base64.b64encode(image).decode('utf-8')
    else:
        raise ValueError("Image must be a file path, base64 string, or bytes")
    
    request = EditImageRequest(
        prompt=prompt,
        image=image_data,
        model=model,
        mask=mask,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        **kwargs
    )
    
    response = await self.client.post_async(
        "/image/edit",
        request.model_dump(exclude_none=True)
    )
    
    return ImageGenerationResponse.model_validate(response)
```

### Task 5: Add Model Validation Decorator

**Important**: Add the `@validate_model_params` decorator to the edit methods like other image methods:

```python
from .validators import validate_model_params

@validate_model_params
def edit(self, prompt: str, image: Union[str, bytes, Path], **kwargs):
    # ... implementation
```

### Task 6: Write Comprehensive Tests

**File**: `tests/test_image.py`

**Add after existing test classes**:

```python
class TestImageEdit:
    """Test image editing functionality."""
    
    @pytest.mark.unit
    def test_edit_image_basic(self):
        """Test basic image editing."""
        with respx.mock:
            # Mock API response
            mock_response = {
                "id": "edit_123",
                "images": [{"url": "https://example.com/edited.jpg"}],
                "created": 1234567890
            }
            
            respx.post("/image/edit").mock(return_value=Response(200, json=mock_response))
            
            client = VeniceClient(api_key="test_key")
            image = Image(client)
            
            # Test with base64 string
            result = image.edit(
                prompt="Make the sky blue",
                image="base64_encoded_image_data"
            )
            
            assert result.id == "edit_123"
            assert len(result.images) == 1
    
    @pytest.mark.unit
    def test_edit_image_with_parameters(self):
        """Test image editing with optional parameters."""
        with respx.mock:
            mock_response = {
                "id": "edit_456",
                "images": [{"url": "https://example.com/edited2.jpg"}],
                "created": 1234567890
            }
            
            respx.post("/image/edit").mock(return_value=Response(200, json=mock_response))
            
            client = VeniceClient(api_key="test_key")
            image = Image(client)
            
            result = image.edit(
                prompt="Change the lighting",
                image="base64_data",
                strength=0.8,
                guidance_scale=7.5,
                num_inference_steps=50,
                seed=42
            )
            
            assert result.id == "edit_456"
    
    @pytest.mark.unit
    def test_edit_image_file_input(self):
        """Test image editing with file input."""
        # Create a temporary image file for testing
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake_image_data")
            tmp_path = tmp.name
        
        try:
            with respx.mock:
                mock_response = {
                    "id": "edit_789",
                    "images": [{"url": "https://example.com/edited3.jpg"}],
                    "created": 1234567890
                }
                
                respx.post("/image/edit").mock(return_value=Response(200, json=mock_response))
                
                client = VeniceClient(api_key="test_key")
                image = Image(client)
                
                result = image.edit(
                    prompt="Add a rainbow",
                    image=tmp_path
                )
                
                assert result.id == "edit_789"
        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)
    
    @pytest.mark.unit
    def test_edit_image_validation_errors(self):
        """Test image editing validation errors."""
        client = VeniceClient(api_key="test_key")
        image = Image(client)
        
        # Test empty prompt
        with pytest.raises(ValidationError):
            image.edit(prompt="", image="base64_data")
        
        # Test invalid strength
        with pytest.raises(ValidationError):
            image.edit(prompt="test", image="base64_data", strength=2.0)
        
        # Test invalid guidance_scale
        with pytest.raises(ValidationError):
            image.edit(prompt="test", image="base64_data", guidance_scale=25.0)
    
    @pytest.mark.integration
    def test_real_image_edit(self):
        """Test real image editing with Venice.ai API."""
        client = VeniceClient()
        image = Image(client)
        
        # Use a simple base64 encoded 1x1 pixel image for testing
        tiny_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        result = image.edit(
            prompt="Make this image blue",
            image=tiny_image
        )
        
        assert result.id is not None
        assert len(result.images) > 0
        assert result.images[0].url is not None
```

### Task 7: Update Documentation and Examples

**File**: `src/example_usage.py`

**Add image editing example**:

```python
# Add to image examples section
print("\\n=== Image Editing ===")
try:
    # Example with base64 image
    result = image.edit(
        prompt="Make the sky more dramatic",
        image="base64_encoded_image_data",
        strength=0.7,
        guidance_scale=8.0
    )
    print(f"Edited image: {result.images[0].url}")
    
    # Example with file input
    result = image.edit(
        prompt="Add autumn colors",
        image="/path/to/input/image.jpg",
        num_inference_steps=30
    )
    print(f"Edited image: {result.images[0].url}")
    
except Exception as e:
    print(f"Image editing error: {e}")
```

### Task 8: Run Comprehensive Testing

```bash
# Run new image edit tests
pytest tests/test_image.py::TestImageEdit -v

# Run all image tests
pytest tests/test_image.py -v

# Run integration tests
pytest tests/test_image.py -m integration -v

# Run full test suite to ensure no regressions
pytest tests/ -m "not integration" --cov=pyvenice --cov-report=term-missing

# Test API contract validation
python scripts/api-contract-validator.py
```

## Success Criteria

Before proceeding to Session 4, verify:

1. **New Endpoint Implemented**:
   - `POST /image/edit` endpoint working in PyVenice
   - Both sync and async versions implemented
   - Proper error handling and validation

2. **Tests Passing**:
   - All new unit tests for image editing pass
   - Integration test for real API editing works
   - All existing image tests continue to pass

3. **Code Quality Maintained**:
   - `ruff check .` passes without errors
   - Test coverage remains at 80%+
   - No regressions in existing functionality

4. **Documentation Updated**:
   - Docstrings added for new methods
   - Example usage provided
   - API specification coverage increased

## Expected File Changes

After successful completion:
- `pyvenice/image.py` - Added EditImageRequest model and edit() methods
- `tests/test_image.py` - Added comprehensive test suite for editing
- `src/example_usage.py` - Added image editing examples

## Handoff to Session 4

**Status**: Image edit endpoint implemented and tested

**Next Session Focus**: Implement missing API key management endpoints

**Key Information for Session 4**:
- New image/edit endpoint is fully functional
- All tests are passing (stable foundation)
- API key management missing: POST /api_keys, DELETE /api_keys, GET /api_keys/generate_web3_key
- Follow patterns from existing api_keys.py implementation

**Files to Focus On**:
- `pyvenice/api_keys.py` - Add create_key(), delete_key(), get_web3_token() methods
- `tests/test_api_keys.py` - Add comprehensive tests for new methods
- `docs/swagger.yaml` - Reference for API key management schemas

## Troubleshooting

**If API specification not found**:
- Check if local swagger.yaml was properly updated in Session 1
- Fetch live API spec manually and search for EditImageRequest
- Check API monitoring output for endpoint details

**If tests fail**:
- Verify API key has sufficient permissions for image editing
- Check if image edit endpoint is available in your API tier
- Validate base64 encoding/decoding of test images

**If integration fails**:
- Test with a minimal 1x1 pixel image first
- Check API response format matches expected schema
- Verify model validation is working correctly

## Validation Commands

```bash
# Final validation before ending session
source .venv/bin/activate

# Test new image edit functionality
pytest tests/test_image.py::TestImageEdit -v | grep -E "(PASSED|FAILED)"

# Test all image functionality  
pytest tests/test_image.py -v | grep -E "(PASSED|FAILED)" | tail -5

# Test integration with real API
pytest tests/test_image.py::TestImageEdit::test_real_image_edit -v

# Test code quality
ruff check . && echo "✅ Code quality maintained"

# Test API contract validation
python scripts/api-contract-validator.py | grep -E "(PASS|FAIL)" | tail -5
```

If all validation commands pass, Session 3 is complete and ready for Session 4.