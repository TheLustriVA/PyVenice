"""
Billing/usage endpoint wrapper for Venice.ai API.
"""

from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field

from .client import BaseResource


class InferenceDetails(BaseModel):
    """Details about inference usage."""

    requestId: Optional[str] = Field(None, description="Request identifier")
    inferenceExecutionTime: Optional[int] = Field(
        None, description="Execution time in milliseconds"
    )
    promptTokens: Optional[int] = Field(None, description="Number of prompt tokens")
    completionTokens: Optional[int] = Field(
        None, description="Number of completion tokens"
    )
    imageCount: Optional[int] = Field(None, description="Number of images generated")
    imageDetails: Optional[Dict[str, Any]] = Field(
        None, description="Image generation details"
    )


class UsageEntry(BaseModel):
    """Single billing usage entry."""

    timestamp: str = Field(description="Timestamp of usage")
    sku: str = Field(description="SKU identifier")
    pricePerUnitUsd: float = Field(description="Price per unit in USD")
    units: float = Field(description="Number of units")
    amount: float = Field(description="Total amount charged")
    currency: Literal["USD", "VCU"] = Field(description="Currency type")
    notes: str = Field(description="Usage notes")
    inferenceDetails: Optional[InferenceDetails] = None


class UsageResponse(BaseModel):
    limit: Optional[int] = Field(None, description="limit")
    page: Optional[int] = Field(None, description="page")
    """Response from billing usage endpoint."""

    data: List[UsageEntry]
    pagination: Dict[str, int]


class Billing(BaseResource):
    """
    Interface for Venice.ai billing/usage endpoint.

    Provides methods to retrieve billing usage data.
    This is a beta endpoint and may be subject to change.
    """

    def get_usage(
        self,
        *,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        currency: Optional[Literal["USD", "VCU"]] = None,
        page: int = 1,
        limit: int = 200,
        sort_order: Literal["asc", "desc"] = "desc",
        format: Literal["json", "csv"] = "json",
    ) -> Union[UsageResponse, str]:
        """
        Get billing usage data.

        Args:
            start_date: Start date for filtering records.
            end_date: End date for filtering records.
            currency: Filter by currency type.
            page: Page number for pagination (starts at 1).
            limit: Number of items per page (max 500).
            sort_order: Sort order for createdAt field.
            format: Response format (json or csv).

        Returns:
            UsageResponse if format is json, CSV string if format is csv.
        """
        params = {
            "page": page,
            "limit": limit,
            "sortOrder": sort_order,
        }

        if start_date:
            if isinstance(start_date, str):
                params["startDate"] = start_date
            else:
                params["startDate"] = start_date.isoformat()
        if end_date:
            if isinstance(end_date, str):
                params["endDate"] = end_date
            else:
                params["endDate"] = end_date.isoformat()
        if currency:
            params["currency"] = currency

        headers = {}
        if format == "csv":
            headers["Accept"] = "text/csv"

        response = self.client._request(
            "GET",
            "/billing/usage",
            params=params,
            headers=headers,
            stream=True,
        )

        if format == "csv":
            return response.text
        else:
            # Extract pagination from headers if available
            pagination = {}
            if hasattr(response, "headers"):
                response_headers = response.headers
                if "x-pagination-page" in response_headers:
                    pagination["page"] = int(response_headers["x-pagination-page"])
                if "x-pagination-limit" in response_headers:
                    pagination["limit"] = int(response_headers["x-pagination-limit"])
                if "x-pagination-total" in response_headers:
                    pagination["total"] = int(response_headers["x-pagination-total"])
                if "x-pagination-total-pages" in response_headers:
                    pagination["total_pages"] = int(
                        response_headers["x-pagination-total-pages"]
                    )

            result = response.json()
            result["pagination"] = pagination
            return UsageResponse(**result)

    async def get_usage_async(
        self,
        *,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        currency: Optional[Literal["USD", "VCU"]] = None,
        page: int = 1,
        limit: int = 200,
        sort_order: Literal["asc", "desc"] = "desc",
        format: Literal["json", "csv"] = "json",
    ) -> Union[UsageResponse, str]:
        """Async version of get_usage()."""
        params = {
            "page": page,
            "limit": limit,
            "sortOrder": sort_order,
        }

        if start_date:
            if isinstance(start_date, str):
                params["startDate"] = start_date
            else:
                params["startDate"] = start_date.isoformat()
        if end_date:
            if isinstance(end_date, str):
                params["endDate"] = end_date
            else:
                params["endDate"] = end_date.isoformat()
        if currency:
            params["currency"] = currency

        headers = {}
        if format == "csv":
            headers["Accept"] = "text/csv"

        response = await self.client._request_async(
            "GET",
            "/billing/usage",
            params=params,
            headers=headers,
            stream=True,
        )

        if format == "csv":
            return response.text
        else:
            # Extract pagination from headers if available
            pagination = {}
            if hasattr(response, "headers"):
                response_headers = response.headers
                if "x-pagination-page" in response_headers:
                    pagination["page"] = int(response_headers["x-pagination-page"])
                if "x-pagination-limit" in response_headers:
                    pagination["limit"] = int(response_headers["x-pagination-limit"])
                if "x-pagination-total" in response_headers:
                    pagination["total"] = int(response_headers["x-pagination-total"])
                if "x-pagination-total-pages" in response_headers:
                    pagination["total_pages"] = int(
                        response_headers["x-pagination-total-pages"]
                    )

            result = response.json()
            result["pagination"] = pagination
            return UsageResponse(**result)

    def get_all_usage(
        self,
        *,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
        currency: Optional[Literal["USD", "VCU"]] = None,
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> List[UsageEntry]:
        """
        Get all billing usage data by paginating through all pages.

        Args:
            start_date: Start date for filtering records.
            end_date: End date for filtering records.
            currency: Filter by currency type.
            sort_order: Sort order for createdAt field.

        Returns:
            List of all usage entries.
        """
        all_entries = []
        page = 1

        while True:
            response = self.get_usage(
                start_date=start_date,
                end_date=end_date,
                currency=currency,
                page=page,
                limit=500,  # Max limit
                sort_order=sort_order,
            )

            all_entries.extend(response.data)

            # Check if there are more pages
            if page >= response.pagination.get("total_pages", 1):
                break

            page += 1

        return all_entries

    def get_usage_summary(
        self,
        *,
        start_date: Optional[Union[datetime, str]] = None,
        end_date: Optional[Union[datetime, str]] = None,
    ) -> Dict[str, Any]:
        """
        Get a summary of usage by model and currency.

        Args:
            start_date: Start date for filtering records.
            end_date: End date for filtering records.

        Returns:
            Dictionary with usage summary by model and currency.
        """
        all_usage = self.get_all_usage(
            start_date=start_date,
            end_date=end_date,
        )

        summary = {
            "total_usd": 0.0,
            "total_vcu": 0.0,
            "by_model": {},
            "by_service": {},
            "entry_count": len(all_usage),
        }

        for entry in all_usage:
            # Update totals
            if entry.currency == "USD":
                summary["total_usd"] += entry.amount
            else:
                summary["total_vcu"] += entry.amount

            # Update by model
            model_id = entry.modelId or "unknown"
            if model_id not in summary["by_model"]:
                summary["by_model"][model_id] = {
                    "usd": 0.0,
                    "vcu": 0.0,
                    "count": 0,
                }
            summary["by_model"][model_id]["count"] += 1
            if entry.currency == "USD":
                summary["by_model"][model_id]["usd"] += entry.amount
            else:
                summary["by_model"][model_id]["vcu"] += entry.amount

            # Update by service
            if entry.service not in summary["by_service"]:
                summary["by_service"][entry.service] = {
                    "usd": 0.0,
                    "vcu": 0.0,
                    "count": 0,
                }
            summary["by_service"][entry.service]["count"] += 1
            if entry.currency == "USD":
                summary["by_service"][entry.service]["usd"] += entry.amount
            else:
                summary["by_service"][entry.service]["vcu"] += entry.amount

        return summary
