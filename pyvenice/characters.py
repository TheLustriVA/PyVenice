"""
Characters endpoint wrapper for Venice.ai API.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from .client import BaseResource, VeniceClient


class Character(BaseModel):
    """Character information."""

    adult: bool = Field(description="Whether the character is adult content")
    createdAt: str = Field(description="Date when character was created")
    description: Optional[str] = Field(None, description="Character description")
    emoji: Optional[str] = Field(None, description="Character emoji")
    hidden: bool = Field(False, description="Whether character is hidden")
    name: str = Field(description="Character name")
    modelId: Optional[str] = Field(None, description="Model ID for this character")
    shareUrl: Optional[str] = Field(None, description="Public sharing URL")
    stats: Optional[dict] = Field(None, description="Character usage statistics")
    webEnabled: Optional[bool] = Field(
        None, description="Whether character is web-enabled"
    )
    id: Optional[str] = Field(None, description="Character ID (alias for slug)")
    slug: str = Field(description="Character slug for API use")
    tags: List[str] = Field(default_factory=list, description="Character tags")
    updatedAt: str = Field(description="Date when character was last updated")


class CharacterListResponse(BaseModel):
    """Response from characters list endpoint."""

    data: List[Character]
    object: Literal["list"] = "list"


class Characters(BaseResource):
    """
    Interface for Venice.ai characters endpoint.

    This is a preview API that provides access to Venice characters
    that can be used with the chat completions endpoint.
    """

    def __init__(self, client: VeniceClient):
        super().__init__(client)
        self._characters_cache: Optional[CharacterListResponse] = None

    def list(self, force_refresh: bool = False) -> CharacterListResponse:
        """
        List available characters.

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            CharacterListResponse with list of characters.
        """
        if not force_refresh and self._characters_cache:
            return self._characters_cache

        response = self.client.get("/characters")
        self._characters_cache = CharacterListResponse(**response)
        return self._characters_cache

    async def list_async(self, force_refresh: bool = False) -> CharacterListResponse:
        """Async version of list()."""
        if not force_refresh and self._characters_cache:
            return self._characters_cache

        response = await self.client.get_async("/characters")
        self._characters_cache = CharacterListResponse(**response)
        return self._characters_cache

    def get_character(
        self, slug: str, force_refresh: bool = False
    ) -> Optional[Character]:
        """
        Get a specific character by slug.

        Args:
            slug: The character slug.
            force_refresh: Force refresh of cached data.

        Returns:
            Character object if found, None otherwise.
        """
        characters = self.list(force_refresh=force_refresh)
        for character in characters.data:
            if character.slug == slug:
                return character
        return None

    def list_by_tag(self, tag: str, force_refresh: bool = False) -> List[Character]:
        """
        List characters filtered by tag.

        Args:
            tag: Tag to filter by.
            force_refresh: Force refresh of cached data.

        Returns:
            List of characters with the specified tag.
        """
        characters = self.list(force_refresh=force_refresh)
        return [char for char in characters.data if tag in char.tags]

    def list_adult_only(self, force_refresh: bool = False) -> List[Character]:
        """
        List adult-only characters.

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            List of adult characters.
        """
        characters = self.list(force_refresh=force_refresh)
        return [char for char in characters.data if char.adult]

    def list_safe(self, force_refresh: bool = False) -> List[Character]:
        """
        List non-adult characters.

        Args:
            force_refresh: Force refresh of cached data.

        Returns:
            List of non-adult characters.
        """
        characters = self.list(force_refresh=force_refresh)
        return [char for char in characters.data if not char.adult]
