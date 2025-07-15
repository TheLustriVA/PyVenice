"""
API keys management endpoints for Venice.ai API.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from .client import BaseResource


class APIKeyInfo(BaseModel):
    """API key information."""

    key: str
    name: Optional[str] = None
    created: datetime
    last_used: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)


class RateLimit(BaseModel):
    """Rate limit information."""

    limit: int
    remaining: int
    reset: datetime
    window: str


class RateLimitLog(BaseModel):
    """Rate limit log entry."""

    timestamp: datetime
    endpoint: str
    status_code: int
    tokens_used: Optional[int] = None
    rate_limit_hit: bool = False


class Web3KeyResponse(BaseModel):
    """Response from web3 key generation."""

    key: str
    address: str
    message: str


class CreateApiKeyRequest(BaseModel):
    """Request model for creating API keys."""

    apiKeyType: Literal["INFERENCE", "ADMIN"] = Field(
        ..., description="The API Key type"
    )
    description: str = Field(..., description="The API Key description")
    consumptionLimit: Dict[str, Optional[float]] = Field(
        ..., description="The API Key consumption limits"
    )
    expiresAt: Optional[str] = Field(None, description="The API Key expiration date")


class CreateApiKeyResponse(BaseModel):
    """Response model for creating API keys."""

    data: Dict[str, Any] = Field(..., description="API key data")


class DeleteApiKeyResponse(BaseModel):
    """Response model for deleting API keys."""

    success: bool = Field(..., description="Success status")


class Web3TokenResponse(BaseModel):
    """Response model for Web3 token."""

    data: Dict[str, Any] = Field(..., description="Web3 token data")
    success: bool = Field(..., description="Success status")


class APIKeys(BaseResource):
    """
    Interface for Venice.ai API key management endpoints.

    Provides methods to:
    - Get API key information
    - Check rate limits
    - View rate limit logs
    - Generate web3 keys
    """

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the current API key.

        Returns:
            Dictionary containing API key information.
        """
        return self.client.get("/api_keys")

    async def get_info_async(self) -> Dict[str, Any]:
        """Async version of get_info()."""
        return await self.client.get_async("/api_keys")

    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get current rate limit status.

        Returns:
            Dictionary containing rate limit information for different endpoints.
        """
        return self.client.get("/api_keys/rate_limits")

    async def get_rate_limits_async(self) -> Dict[str, Any]:
        """Async version of get_rate_limits()."""
        return await self.client.get_async("/api_keys/rate_limits")

    def get_rate_limit_log(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get rate limit log entries.

        Args:
            limit: Maximum number of entries to return.
            offset: Number of entries to skip.
            start_date: Filter logs after this date.
            end_date: Filter logs before this date.

        Returns:
            Dictionary containing log entries and metadata.
        """
        params = {
            "limit": limit,
            "offset": offset,
        }

        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        return self.client.get("/api_keys/rate_limits/log", params=params)

    async def get_rate_limit_log_async(
        self,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Async version of get_rate_limit_log()."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        return await self.client.get_async("/api_keys/rate_limits/log", params=params)

    def generate_web3_key(
        self,
        wallet_address: str,
        signature: str,
        message: str,
    ) -> Web3KeyResponse:
        """
        Generate a web3 API key.

        Args:
            wallet_address: Ethereum wallet address.
            signature: Signed message from the wallet.
            message: The message that was signed.

        Returns:
            Web3KeyResponse containing the generated key.
        """
        payload = {
            "wallet_address": wallet_address,
            "signature": signature,
            "message": message,
        }

        response = self.client.post("/api_keys/generate_web3_key", payload)
        return Web3KeyResponse(**response)

    async def generate_web3_key_async(
        self,
        wallet_address: str,
        signature: str,
        message: str,
    ) -> Web3KeyResponse:
        """Async version of generate_web3_key()."""
        payload = {
            "wallet_address": wallet_address,
            "signature": signature,
            "message": message,
        }

        response = await self.client.post_async("/api_keys/generate_web3_key", payload)
        return Web3KeyResponse(**response)

    def create_key(
        self,
        *,
        key_type: Literal["INFERENCE", "ADMIN"],
        description: str,
        usd_limit: Optional[float] = None,
        vcu_limit: Optional[float] = None,
        diem_limit: Optional[float] = None,
        expires_at: Optional[str] = None,
    ) -> CreateApiKeyResponse:
        """
        Create a new API key.

        Args:
            key_type: The API key type (INFERENCE or ADMIN)
            description: Description for the API key
            usd_limit: USD consumption limit (optional)
            vcu_limit: VCU consumption limit (optional)
            diem_limit: Diem consumption limit (optional)
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
        if diem_limit is not None:
            consumption_limits["diem"] = diem_limit

        # Create request
        request = CreateApiKeyRequest(
            apiKeyType=key_type,
            description=description,
            consumptionLimit=consumption_limits,
            expiresAt=expires_at,
        )

        # Make API call
        response = self.client.post("/api_keys", request.model_dump(exclude_none=True))

        return CreateApiKeyResponse.model_validate(response)

    async def create_key_async(
        self,
        *,
        key_type: Literal["INFERENCE", "ADMIN"],
        description: str,
        usd_limit: Optional[float] = None,
        vcu_limit: Optional[float] = None,
        diem_limit: Optional[float] = None,
        expires_at: Optional[str] = None,
    ) -> CreateApiKeyResponse:
        """Async version of create_key()."""

        # Build consumption limits
        consumption_limits = {}
        if usd_limit is not None:
            consumption_limits["usd"] = usd_limit
        if vcu_limit is not None:
            consumption_limits["vcu"] = vcu_limit
        if diem_limit is not None:
            consumption_limits["diem"] = diem_limit

        request = CreateApiKeyRequest(
            apiKeyType=key_type,
            description=description,
            consumptionLimit=consumption_limits,
            expiresAt=expires_at,
        )

        response = await self.client.post_async(
            "/api_keys", request.model_dump(exclude_none=True)
        )

        return CreateApiKeyResponse.model_validate(response)

    def delete_key(self, key_id: str) -> DeleteApiKeyResponse:
        """
        Delete an API key.

        Args:
            key_id: The ID of the API key to delete

        Returns:
            DeleteApiKeyResponse with deletion result

        Raises:
            VeniceAPIError: If the API request fails
        """

        # Make API call with query parameter
        response = self.client._request("DELETE", "/api_keys", params={"id": key_id})

        return DeleteApiKeyResponse.model_validate(response)

    async def delete_key_async(self, key_id: str) -> DeleteApiKeyResponse:
        """Async version of delete_key()."""

        response = await self.client._request_async(
            "DELETE", "/api_keys", params={"id": key_id}
        )

        return DeleteApiKeyResponse.model_validate(response)

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
