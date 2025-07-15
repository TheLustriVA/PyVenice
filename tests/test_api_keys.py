"""Unit tests for the API Keys module."""

import pytest
import respx
from httpx import Response

from pyvenice.api_keys import APIKeys
from pyvenice.client import VeniceClient
from pydantic import ValidationError


@pytest.mark.unit
class TestAPIKeys:
    """Test APIKeys class functionality."""

    def test_api_keys_init(self, client):
        """Test APIKeys initialization."""
        api_keys = APIKeys(client)
        assert api_keys.client == client

    @respx.mock
    def test_get_info(self, client):
        """Test get_info method."""
        mock_response = {
            "data": [
                {
                    "apiKeyType": "INFERENCE",
                    "description": "Test key",
                    "id": "key_123"
                }
            ]
        }
        
        respx.get("https://api.venice.ai/api/v1/api_keys").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        result = api_keys.get_info()
        
        assert result["data"][0]["apiKeyType"] == "INFERENCE"
        assert result["data"][0]["description"] == "Test key"

    @respx.mock
    def test_get_rate_limits(self, client):
        """Test get_rate_limits method."""
        mock_response = {
            "data": {
                "chat": {"limit": 100, "remaining": 95, "reset": 1234567890},
                "embeddings": {"limit": 50, "remaining": 45, "reset": 1234567890}
            }
        }
        
        respx.get("https://api.venice.ai/api/v1/api_keys/rate_limits").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        result = api_keys.get_rate_limits()
        
        assert result["data"]["chat"]["limit"] == 100
        assert result["data"]["embeddings"]["remaining"] == 45


@pytest.mark.unit 
class TestApiKeyManagement:
    """Test API key management functionality."""
    
    @respx.mock
    def test_create_api_key_basic(self, client):
        """Test basic API key creation."""
        mock_response = {
            "data": {
                "apiKey": "test_key_123",
                "apiKeyType": "INFERENCE",
                "description": "Test key",
                "id": "key_id_123"
            }
        }
        
        respx.post("https://api.venice.ai/api/v1/api_keys").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = api_keys.create_key(
            key_type="INFERENCE",
            description="Test key",
            usd_limit=10.0,
            vcu_limit=100.0
        )
        
        assert result.data["apiKey"] == "test_key_123"
        assert result.data["apiKeyType"] == "INFERENCE"
        assert result.data["description"] == "Test key"
    
    @respx.mock
    def test_create_api_key_admin(self, client):
        """Test ADMIN API key creation."""
        mock_response = {
            "data": {
                "apiKey": "admin_key_456",
                "apiKeyType": "ADMIN",
                "description": "Admin key",
                "id": "admin_id_456"
            }
        }
        
        respx.post("https://api.venice.ai/api/v1/api_keys").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = api_keys.create_key(
            key_type="ADMIN",
            description="Admin key",
            expires_at="2025-12-31T23:59:59Z"
        )
        
        assert result.data["apiKeyType"] == "ADMIN"
        assert result.data["description"] == "Admin key"
    
    @respx.mock
    def test_create_api_key_all_limits(self, client):
        """Test API key creation with all consumption limits."""
        mock_response = {
            "data": {
                "apiKey": "full_key_789",
                "apiKeyType": "INFERENCE",
                "description": "Full limits key",
                "id": "full_id_789",
                "consumptionLimit": {
                    "usd": 25.0,
                    "vcu": 250.0,
                    "diem": 5.0
                }
            }
        }
        
        respx.post("https://api.venice.ai/api/v1/api_keys").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = api_keys.create_key(
            key_type="INFERENCE",
            description="Full limits key",
            usd_limit=25.0,
            vcu_limit=250.0,
            diem_limit=5.0
        )
        
        assert result.data["consumptionLimit"]["usd"] == 25.0
        assert result.data["consumptionLimit"]["vcu"] == 250.0
        assert result.data["consumptionLimit"]["diem"] == 5.0
    
    def test_create_api_key_validation(self, client):
        """Test API key creation validation."""
        api_keys = APIKeys(client)
        
        # Test invalid key type
        with pytest.raises(ValidationError):
            api_keys.create_key(
                key_type="INVALID",  # This should fail
                description="Test key"
            )
    
    @respx.mock
    def test_delete_api_key(self, client):
        """Test API key deletion."""
        mock_response = {"success": True, "message": "API key deleted"}
        
        respx.delete("https://api.venice.ai/api/v1/api_keys").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = api_keys.delete_key("key_id_123")
        
        assert result.success is True
    
    @respx.mock
    def test_get_web3_token(self, client):
        """Test Web3 token retrieval."""
        mock_response = {
            "data": {
                "token": "jwt_token_here"
            },
            "success": True
        }
        
        respx.get("https://api.venice.ai/api/v1/api_keys/generate_web3_key").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = api_keys.get_web3_token()
        
        assert result.success is True
        assert result.data["token"] == "jwt_token_here"


@pytest.mark.integration
class TestApiKeyIntegration:
    """Integration tests for API key management."""
    
    def test_real_get_info(self):
        """Test real API key info retrieval."""
        client = VeniceClient()
        api_keys = APIKeys(client)
        
        result = api_keys.get_info()
        
        assert "data" in result
        assert isinstance(result["data"], list)
    
    def test_real_get_rate_limits(self):
        """Test real rate limits retrieval."""
        client = VeniceClient()
        api_keys = APIKeys(client)
        
        result = api_keys.get_rate_limits()
        
        assert "data" in result
        assert isinstance(result["data"], dict)
    
    def test_real_web3_token_retrieval(self):
        """Test real Web3 token retrieval."""
        client = VeniceClient()
        api_keys = APIKeys(client)
        
        result = api_keys.get_web3_token()
        
        assert result.success is True
        assert "token" in result.data
        assert result.data["token"] is not None
    
    @pytest.mark.skip(reason="Requires ADMIN privileges and creates real API keys")
    def test_real_api_key_creation_and_deletion(self):
        """Test real API key creation and deletion."""
        client = VeniceClient()
        api_keys = APIKeys(client)
        
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
        assert delete_result.success is True


@pytest.mark.asyncio
class TestApiKeyAsync:
    """Test async API key methods."""
    
    @respx.mock
    async def test_create_key_async(self, client):
        """Test async API key creation."""
        mock_response = {
            "data": {
                "apiKey": "async_key_123",
                "apiKeyType": "INFERENCE",
                "description": "Async test key",
                "id": "async_id_123"
            }
        }
        
        respx.post("https://api.venice.ai/api/v1/api_keys").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = await api_keys.create_key_async(
            key_type="INFERENCE",
            description="Async test key",
            usd_limit=5.0
        )
        
        assert result.data["apiKey"] == "async_key_123"
        assert result.data["description"] == "Async test key"
    
    @respx.mock
    async def test_delete_key_async(self, client):
        """Test async API key deletion."""
        mock_response = {"success": True}
        
        respx.delete("https://api.venice.ai/api/v1/api_keys").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = await api_keys.delete_key_async("async_key_123")
        
        assert result.success is True
    
    @respx.mock
    async def test_get_web3_token_async(self, client):
        """Test async Web3 token retrieval."""
        mock_response = {
            "data": {"token": "async_jwt_token"},
            "success": True
        }
        
        respx.get("https://api.venice.ai/api/v1/api_keys/generate_web3_key").mock(
            return_value=Response(200, json=mock_response)
        )
        
        api_keys = APIKeys(client)
        
        result = await api_keys.get_web3_token_async()
        
        assert result.success is True
        assert result.data["token"] == "async_jwt_token"