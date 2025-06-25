"""Unit tests for the Billing module."""

import pytest
import httpx
import respx

from pyvenice.billing import Billing
from pyvenice.client import VeniceClient
from pyvenice.exceptions import InvalidRequestError


@pytest.mark.unit
class TestBilling:
    """Test Billing class functionality."""

    def test_billing_init(self, client):
        """Test Billing initialization."""
        billing = Billing(client)
        assert billing.client == client

    @respx.mock
    def test_get_usage_basic(self, respx_mock, mock_billing_response, client):
        """Test basic usage retrieval."""
        respx_mock.get("https://api.venice.ai/api/v1/billing/usage").mock(
            return_value=httpx.Response(
                200,
                json=mock_billing_response,
                headers={
                    "x-pagination-page": "1",
                    "x-pagination-limit": "200",
                    "x-pagination-total": "1",
                    "x-pagination-total-pages": "1",
                },
            )
        )

        billing = Billing(client)
        response = billing.get_usage()

        assert len(response.data) == 1
        assert response.data[0].amount == 0.001
        assert response.data[0].currency == "USD"
        assert response.data[0].sku == "venice-uncensored-llm-output-mtoken"
        assert response.pagination["total"] == 1

    @respx.mock
    def test_get_usage_with_date_range(self, respx_mock, mock_billing_response, client):
        """Test usage retrieval with date range."""
        respx_mock.get("https://api.venice.ai/api/v1/billing/usage").mock(
            return_value=httpx.Response(
                200,
                json=mock_billing_response,
                headers={
                    "x-pagination-page": "1",
                    "x-pagination-limit": "200",
                    "x-pagination-total": "1",
                    "x-pagination-total-pages": "1",
                },
            )
        )

        from datetime import datetime

        billing = Billing(client)
        response = billing.get_usage(
            start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 2)
        )

        # Verify request was made with correct parameters
        request = respx_mock.calls[0].request
        assert "startDate=2024-01-01T00%3A00%3A00" in str(request.url)
        assert "endDate=2024-01-02T00%3A00%3A00" in str(request.url)

        assert len(response.data) == 1
        assert response.data[0].timestamp == "2024-12-20T21:28:08.934Z"

    @respx.mock
    def test_get_usage_with_pagination(self, respx_mock, mock_billing_response, client):
        """Test usage retrieval with pagination."""
        respx_mock.get("https://api.venice.ai/api/v1/billing/usage").mock(
            return_value=httpx.Response(
                200,
                json=mock_billing_response,
                headers={
                    "x-pagination-page": "1",
                    "x-pagination-limit": "10",
                    "x-pagination-total": "100",
                    "x-pagination-total-pages": "10",
                },
            )
        )

        billing = Billing(client)
        response = billing.get_usage(limit=10, page=2)

        # Verify request was made with correct parameters
        request = respx_mock.calls[0].request
        assert "limit=10" in str(request.url)
        assert "page=2" in str(request.url)

        assert response.pagination["total_pages"] == 10

    @pytest.mark.asyncio
    @respx.mock
    async def test_get_usage_async(self, respx_mock, client):
        """Test async usage retrieval."""
        mock_response = {
            "data": [
                {
                    "timestamp": "2024-01-01T12:00:00Z",
                    "sku": "chat-completion-sku",
                    "pricePerUnitUsd": 1.0,
                    "units": 1.5,
                    "amount": 1.50,
                    "currency": "USD",
                    "notes": "API Inference",
                    "inferenceDetails": {
                        "requestId": "test-inference-123",
                        "inferenceExecutionTime": 500,
                    },
                }
            ],
            "pagination": {"page": 1, "limit": 200, "total": 1, "total_pages": 1},
        }

        respx_mock.get("https://api.venice.ai/api/v1/billing/usage").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

        billing = Billing(client)
        response = await billing.get_usage_async()

        assert len(response.data) == 1
        assert response.data[0].amount == 1.50


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in billing."""

    @respx.mock
    def test_unauthorized_error(self, respx_mock, client):
        """Test error when unauthorized."""
        error_response = {"error": "Unauthorized access to billing data"}
        respx_mock.get("https://api.venice.ai/api/v1/billing/usage").mock(
            return_value=httpx.Response(401, json=error_response)
        )

        billing = Billing(client)
        with pytest.raises(Exception):  # Will be AuthenticationError
            billing.get_usage()

    @respx.mock
    def test_invalid_date_format_error(self, respx_mock, client):
        """Test error when invalid date format."""
        error_response = {
            "error": "Invalid date format",
            "details": {
                "start_date": {"_errors": ["Date must be in YYYY-MM-DD format"]}
            },
        }
        respx_mock.get("https://api.venice.ai/api/v1/billing/usage").mock(
            return_value=httpx.Response(400, json=error_response)
        )

        billing = Billing(client)
        with pytest.raises(InvalidRequestError):
            billing.get_usage(start_date="invalid-date")


@pytest.mark.integration
class TestBillingIntegration:
    """Integration tests for Billing (requires API key)."""

    def test_real_usage_retrieval(self, skip_if_no_admin_key, integration_admin_key):
        """Test real usage data retrieval."""
        client = VeniceClient(api_key=integration_admin_key)
        billing = Billing(client)

        response = billing.get_usage(limit=5)

        assert hasattr(response, "data")
        assert hasattr(response, "pagination")
        assert isinstance(response.data, list)

        # If there's data, check its structure
        if response.data:
            usage_item = response.data[0]
            assert hasattr(usage_item, "amount")
            assert hasattr(usage_item, "currency")
            assert hasattr(usage_item, "timestamp")
            assert hasattr(usage_item, "sku")
