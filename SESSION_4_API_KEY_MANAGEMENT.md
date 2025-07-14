# Session 4: API Key Management Endpoints

## Context from Previous Sessions

**Session 1 Results**: Foundation dependencies resolved, API specification updated to v20250713.224148
**Session 2 Results**: Schema fixes completed, integration tests passing  
**Session 3 Results**: Image edit endpoint implemented and tested

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

# Verify image edit endpoint is working
pytest tests/test_image.py::TestImageEdit::test_real_image_edit -v | grep PASSED && echo "✅ Image edit implemented"
```

## Goals

1. **Implement Missing API Key Endpoints**: Add POST /api_keys, DELETE /api_keys, GET /api_keys/generate_web3_key
2. **Add Request/Response Models**: Create proper Pydantic models for API key operations
3. **Extend ApiKeys Class**: Add create_key(), delete_key(), get_web3_token() methods
4. **Comprehensive Testing**: Write unit and integration tests for all new endpoints
5. **Error Handling**: Proper error handling for API key operations

## Context from API Audit

### Missing API Key Endpoints:
1. **POST /api_keys** - Create new API key
2. **DELETE /api_keys** - Delete existing API key
3. **GET /api_keys/generate_web3_key** - Get Web3 token for wallet authentication

### Currently Implemented:
- ✅ GET /api_keys - Get API key information
- ✅ GET /api_keys/rate_limits - Get rate limit status
- ✅ GET /api_keys/rate_limits/log - Get rate limit logs
- ✅ POST /api_keys/generate_web3_key - Generate Web3 API key

## Detailed Tasks

### Task 1: Analyze API Specification for Missing Endpoints

```bash
# Check API specification for API key endpoints
grep -n -A 30 "/api_keys:" docs/swagger.yaml
grep -n -A 20 "post:" docs/swagger.yaml | grep -A 20 "api_keys"
grep -n -A 20 "delete:" docs/swagger.yaml | grep -A 20 "api_keys"

# Look for API key related schemas
grep -n -A 10 "ApiKey" docs/swagger.yaml
grep -n -A 10 "CreateApiKey" docs/swagger.yaml
grep -n -A 10 "DeleteApiKey" docs/swagger.yaml
```

### Task 2: Study Existing ApiKeys Module Pattern

**File**: `pyvenice/api_keys.py`

```bash
# Examine existing api_keys.py structure
head -50 pyvenice/api_keys.py

# Look at existing methods
grep -n "def " pyvenice/api_keys.py

# Study the existing pattern for generate_web3_key
grep -n -A 15 "def generate_web3_key" pyvenice/api_keys.py
```

### Task 3: Add Request/Response Models

**File**: `pyvenice/api_keys.py`

**Add after existing imports**:

```python
# Add these models at the top of the file after existing imports
class CreateApiKeyRequest(BaseModel):
    """Request model for creating API keys."""
    
    apiKeyType: Literal["INFERENCE", "ADMIN"] = Field(
        ..., description="The API Key type"
    )
    description: str = Field(..., description="The API Key description")
    consumptionLimits: Dict[str, Optional[float]] = Field(
        ..., description="The API Key consumption limits"
    )
    expiresAt: Optional[str] = Field(
        None, description="The API Key expiration date"
    )

class CreateApiKeyResponse(BaseModel):
    """Response model for creating API keys."""
    
    data: Dict[str, Any] = Field(..., description="API key data")
    
class DeleteApiKeyRequest(BaseModel):
    """Request model for deleting API keys."""
    
    id: str = Field(..., description="The ID of the API key to delete")

class Web3TokenResponse(BaseModel):
    """Response model for Web3 token."""
    
    data: Dict[str, Any] = Field(..., description="Web3 token data")
    success: bool = Field(..., description="Success status")
```

### Task 4: Add create_key() Method

**File**: `pyvenice/api_keys.py`

**Add after existing methods in ApiKeys class**:

```python
def create_key(
    self,
    *,
    key_type: Literal["INFERENCE", "ADMIN"],
    description: str,
    usd_limit: Optional[float] = None,
    vcu_limit: Optional[float] = None,
    expires_at: Optional[str] = None
) -> CreateApiKeyResponse:
    """
    Create a new API key.
    
    Args:
        key_type: The API key type (INFERENCE or ADMIN)
        description: Description for the API key
        usd_limit: USD consumption limit (optional)
        vcu_limit: VCU consumption limit (optional)
        expires_at: Expiration date (optional)
        
    Returns:
        CreateApiKeyResponse with the new API key data
        
    Raises:
        VeniceAPIError: If the API request fails
    """
    
    # Build consumption limits
    consumption_limits = {}
    if usd_limit is not None:
        consumption_limits["usd"] = usd_limit
    if vcu_limit is not None:
        consumption_limits["vcu"] = vcu_limit
    
    # Create request
    request = CreateApiKeyRequest(
        apiKeyType=key_type,
        description=description,
        consumptionLimits=consumption_limits,
        expiresAt=expires_at
    )
    
    # Make API call
    response = self.client.post(
        "/api_keys",
        request.model_dump(exclude_none=True)
    )
    
    return CreateApiKeyResponse.model_validate(response)

async def create_key_async(
    self,
    *,
    key_type: Literal["INFERENCE", "ADMIN"],
    description: str,
    usd_limit: Optional[float] = None,
    vcu_limit: Optional[float] = None,
    expires_at: Optional[str] = None
) -> CreateApiKeyResponse:
    """Async version of create_key()."""
    
    # Build consumption limits
    consumption_limits = {}
    if usd_limit is not None:
        consumption_limits["usd"] = usd_limit
    if vcu_limit is not None:
        consumption_limits["vcu"] = vcu_limit
    
    request = CreateApiKeyRequest(
        apiKeyType=key_type,
        description=description,
        consumptionLimits=consumption_limits,
        expiresAt=expires_at
    )
    
    response = await self.client.post_async(
        "/api_keys",
        request.model_dump(exclude_none=True)
    )
    
    return CreateApiKeyResponse.model_validate(response)
```

### Task 5: Add delete_key() Method

**File**: `pyvenice/api_keys.py`

**Add after create_key methods**:

```python
def delete_key(self, key_id: str) -> Dict[str, Any]:
    """
    Delete an API key.
    
    Args:
        key_id: The ID of the API key to delete
        
    Returns:
        Dict with deletion result
        
    Raises:
        VeniceAPIError: If the API request fails
    """
    
    # Make API call with query parameter
    response = self.client._request(
        "DELETE",
        "/api_keys",
        params={"id": key_id}
    )
    
    return response

async def delete_key_async(self, key_id: str) -> Dict[str, Any]:
    """Async version of delete_key()."""
    
    response = await self.client._request_async(
        "DELETE",
        "/api_keys",
        params={"id": key_id}
    )
    
    return response
```

### Task 6: Add get_web3_token() Method

**File**: `pyvenice/api_keys.py`

**Add after delete_key methods**:

```python
def get_web3_token(self) -> Web3TokenResponse:
    """
    Get the token required to generate an API key via a wallet.
    
    Returns:
        Web3TokenResponse with the token data
        
    Raises:
        VeniceAPIError: If the API request fails
    """
    
    response = self.client.get("/api_keys/generate_web3_key")
    return Web3TokenResponse.model_validate(response)

async def get_web3_token_async(self) -> Web3TokenResponse:
    """Async version of get_web3_token()."""
    
    response = await self.client.get_async("/api_keys/generate_web3_key")
    return Web3TokenResponse.model_validate(response)
```

### Task 7: Write Comprehensive Tests

**File**: `tests/test_api_keys.py`

**Add after existing test classes**:

```python
class TestApiKeyManagement:
    """Test API key management functionality."""
    
    @pytest.mark.unit
    def test_create_api_key_basic(self):
        """Test basic API key creation."""
        with respx.mock:
            mock_response = {
                "data": {
                    "apiKey": "test_key_123",
                    "apiKeyType": "INFERENCE",
                    "description": "Test key",
                    "id": "key_id_123"
                }
            }
            
            respx.post("/api_keys").mock(return_value=Response(200, json=mock_response))
            
            client = VeniceClient(api_key="test_key")
            api_keys = ApiKeys(client)
            
            result = api_keys.create_key(
                key_type="INFERENCE",
                description="Test key",
                usd_limit=10.0,
                vcu_limit=100.0
            )
            
            assert result.data["apiKey"] == "test_key_123"
            assert result.data["apiKeyType"] == "INFERENCE"
    
    @pytest.mark.unit
    def test_create_api_key_admin(self):
        """Test ADMIN API key creation."""
        with respx.mock:
            mock_response = {
                "data": {
                    "apiKey": "admin_key_456",
                    "apiKeyType": "ADMIN",
                    "description": "Admin key",
                    "id": "admin_id_456"
                }
            }
            
            respx.post("/api_keys").mock(return_value=Response(200, json=mock_response))
            
            client = VeniceClient(api_key="test_key")
            api_keys = ApiKeys(client)
            
            result = api_keys.create_key(
                key_type="ADMIN",
                description="Admin key",
                expires_at="2025-12-31T23:59:59Z"
            )
            
            assert result.data["apiKeyType"] == "ADMIN"
    
    @pytest.mark.unit
    def test_create_api_key_validation(self):
        """Test API key creation validation."""
        client = VeniceClient(api_key="test_key")
        api_keys = ApiKeys(client)
        
        # Test invalid key type
        with pytest.raises(ValidationError):
            api_keys.create_key(
                key_type="INVALID",
                description="Test key"
            )
        
        # Test empty description
        with pytest.raises(ValidationError):
            api_keys.create_key(
                key_type="INFERENCE",
                description=""
            )
    
    @pytest.mark.unit
    def test_delete_api_key(self):
        """Test API key deletion."""
        with respx.mock:
            mock_response = {"success": True, "message": "API key deleted"}
            
            respx.delete("/api_keys").mock(return_value=Response(200, json=mock_response))
            
            client = VeniceClient(api_key="test_key")
            api_keys = ApiKeys(client)
            
            result = api_keys.delete_key("key_id_123")
            
            assert result["success"] is True
    
    @pytest.mark.unit
    def test_get_web3_token(self):
        """Test Web3 token retrieval."""
        with respx.mock:
            mock_response = {
                "data": {
                    "token": "jwt_token_here"
                },
                "success": True
            }
            
            respx.get("/api_keys/generate_web3_key").mock(return_value=Response(200, json=mock_response))
            
            client = VeniceClient(api_key="test_key")
            api_keys = ApiKeys(client)
            
            result = api_keys.get_web3_token()
            
            assert result.success is True
            assert "token" in result.data
    
    @pytest.mark.integration
    def test_real_api_key_creation(self):
        """Test real API key creation with Venice.ai API."""
        client = VeniceClient()
        api_keys = ApiKeys(client)
        
        # Create a test API key
        result = api_keys.create_key(
            key_type="INFERENCE",
            description="PyVenice Integration Test Key",
            usd_limit=1.0,
            vcu_limit=10.0
        )
        
        assert result.data["apiKey"] is not None
        assert result.data["apiKeyType"] == "INFERENCE"
        assert result.data["description"] == "PyVenice Integration Test Key"
        
        # Clean up - delete the test key
        key_id = result.data["id"]
        delete_result = api_keys.delete_key(key_id)
        assert delete_result is not None
    
    @pytest.mark.integration
    def test_real_web3_token_retrieval(self):
        """Test real Web3 token retrieval."""
        client = VeniceClient()
        api_keys = ApiKeys(client)
        
        result = api_keys.get_web3_token()
        
        assert result.success is True
        assert "token" in result.data
        assert result.data["token"] is not None
```

### Task 8: Update Documentation and Examples

**File**: `src/example_usage.py`

**Add API key management examples**:

```python
# Add to examples
print("\\n=== API Key Management ===")
try:
    # Create a new API key
    result = api_keys.create_key(
        key_type="INFERENCE",
        description="Example API Key",
        usd_limit=10.0,
        vcu_limit=100.0
    )
    print(f"Created API key: {result.data['id']}")
    
    # Get Web3 token
    web3_token = api_keys.get_web3_token()
    print(f"Web3 token available: {web3_token.success}")
    
    # Delete the API key (cleanup)
    delete_result = api_keys.delete_key(result.data['id'])
    print(f"Deleted API key: {delete_result.get('success', False)}")
    
except Exception as e:
    print(f"API key management error: {e}")
```

### Task 9: Run Comprehensive Testing

```bash
# Run new API key management tests
pytest tests/test_api_keys.py::TestApiKeyManagement -v

# Run all API key tests
pytest tests/test_api_keys.py -v

# Run integration tests
pytest tests/test_api_keys.py -m integration -v

# Run full test suite to ensure no regressions
pytest tests/ -m "not integration" --cov=pyvenice --cov-report=term-missing

# Test API contract validation
python scripts/api-contract-validator.py
```

## Success Criteria

Before proceeding to Session 5, verify:

1. **All Missing Endpoints Implemented**:
   - POST /api_keys (create_key) working
   - DELETE /api_keys (delete_key) working  
   - GET /api_keys/generate_web3_key (get_web3_token) working
   - Both sync and async versions implemented

2. **Tests Passing**:
   - All new unit tests for API key management pass
   - Integration tests for real API operations work
   - All existing API key tests continue to pass

3. **Code Quality Maintained**:
   - `ruff check .` passes without errors
   - Test coverage remains at 80%+
   - No regressions in existing functionality

4. **API Coverage Complete**:
   - All 20 Venice.ai API endpoints now implemented
   - 100% endpoint coverage achieved

## Expected File Changes

After successful completion:
- `pyvenice/api_keys.py` - Added create_key(), delete_key(), get_web3_token() methods and models
- `tests/test_api_keys.py` - Added comprehensive test suite for new methods
- `src/example_usage.py` - Added API key management examples

## Handoff to Session 5

**Status**: API key management endpoints implemented, 100% API coverage achieved

**Next Session Focus**: Monitoring system improvements and operational enhancements

**Key Information for Session 5**:
- All API endpoints are now implemented (20/20)
- All integration tests should be passing
- Focus shifts to operational improvements
- Enable automated deployment and monitoring

**Files to Focus On**:
- `scripts/daily-monitor.sh` - Set up automated monitoring
- `scripts/safe-auto-deploy.py` - Enable automated deployment
- Documentation updates and final testing

## Troubleshooting

**If API key creation fails**:
- Check if your current API key has ADMIN privileges
- Verify consumption limits are valid numbers
- Check date format for expires_at parameter

**If deletion fails**:
- Ensure the key ID exists and is valid
- Check if the key belongs to your account
- Verify API key has deletion permissions

**If Web3 token fails**:
- This endpoint typically doesn't require authentication
- Check if the endpoint is available in your region
- Verify network connectivity

## Validation Commands

```bash
# Final validation before ending session
source .venv/bin/activate

# Test new API key management functionality
pytest tests/test_api_keys.py::TestApiKeyManagement -v | grep -E "(PASSED|FAILED)"

# Test all API key functionality
pytest tests/test_api_keys.py -v | grep -E "(PASSED|FAILED)" | tail -5

# Test integration with real API
pytest tests/test_api_keys.py::TestApiKeyManagement::test_real_api_key_creation -v
pytest tests/test_api_keys.py::TestApiKeyManagement::test_real_web3_token_retrieval -v

# Test code quality
ruff check . && echo "✅ Code quality maintained"

# Test API contract validation
python scripts/api-contract-validator.py | grep -E "(PASS|FAIL)" | tail -5

# Verify 100% endpoint coverage
python -c "
import yaml
with open('docs/swagger.yaml') as f:
    spec = yaml.safe_load(f)
    
endpoints = 0
for path, methods in spec.get('paths', {}).items():
    endpoints += len([m for m in methods if m in ['get', 'post', 'put', 'delete']])
    
print(f'Total API endpoints: {endpoints}')
print('Expected: 20 endpoints implemented')
"
```

If all validation commands pass, Session 4 is complete and ready for Session 5.