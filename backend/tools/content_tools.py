"""LiveKit data-channel tools for campus events, competitions, posts, and maps."""

from __future__ import annotations

import json
from typing import Callable, Optional

from livekit import rtc
from livekit.agents import RunContext


class ContentTools:
    def __init__(
        self,
        image_manager,
        image_server,
        room_provider: Callable[[], Optional[rtc.Room]],
        map_navigator=None,
    ):
        self.image_manager = image_manager
        self.image_server = image_server
        self._room_provider = room_provider
        self.map_navigator = map_navigator

    @property
    def room(self) -> Optional[rtc.Room]:
        return self._room_provider()

    async def _publish_image(
        self,
        *,
        category: str,
        image_path,
        caption: str,
        frontend_category: str | None = None,
    ) -> None:
        if not self.room:
            raise RuntimeError("not connected to a room")

        image_url = self.image_server.get_image_url(category, image_path.name)
        payload = {
            "type": "image",
            "category": frontend_category or category.rstrip("s"),
            "url": image_url,
            "caption": caption,
        }
        try:
            image_base64 = self.image_manager.encode_image(image_path)
            payload["base64"] = f"data:image/jpeg;base64,{image_base64}"
        except Exception:
            pass

        await self.room.local_participant.publish_data(json.dumps(payload).encode())

    async def list_available_events(self, context: RunContext) -> str:
        news = self.image_manager.list_campus_news()
        parts: list[str] = []
        for category, names in news.items():
            if names:
                parts.append(f"{category}: {', '.join(names)}")
        if not parts:
            return "There are no campus posters at the moment."
        return "Here's what's on campus — " + "; ".join(parts) + "."

    async def show_event_poster(self, event_description: str, context: RunContext) -> str:
        return await self._show_poster(event_description, category=None, label="event")

    async def show_competition_poster(
        self, competition_description: str, context: RunContext
    ) -> str:
        return await self._show_poster(
            competition_description, category="competitions", label="competition"
        )

    async def show_campus_post(self, post_description: str, context: RunContext) -> str:
        return await self._show_poster(post_description, category="posts", label="post")

    async def _show_poster(
        self,
        description: str,
        *,
        category: str | None,
        label: str,
    ) -> str:
        print(f"Showing {label} poster for: {description}")
        if not self.room:
            return "I am not connected to a room right now."

        image_path, matched_category = self.image_manager.find_poster(description, category)
        if not image_path or not matched_category:
            available = self.image_manager.list_available_events()
            return (
                f"Sorry, I couldn't find a poster for '{description}'. "
                f"We have: {', '.join(available) if available else 'nothing uploaded yet'}."
            )

        frontend_category = {
            "events": "event",
            "competitions": "competition",
            "posts": "post",
        }.get(matched_category, "event")

        await self._publish_image(
            category=matched_category,
            image_path=image_path,
            caption=f"{label.title()}: {description}",
            frontend_category=frontend_category,
        )
        return f"I've displayed the {description} poster for you."

    async def show_location_map(self, location_query: str, context: RunContext) -> str:
        print(f"Showing location map for: {location_query}")
        if not self.room:
            return "I am not connected to a room right now."

        image_path = self.image_manager.find_location_map(location_query)
        if image_path:
            await self._publish_image(
                category="maps",
                image_path=image_path,
                caption=f"Location: {location_query}",
                frontend_category="map",
            )
            extra = ""
            if self.map_navigator and self.map_navigator.available:
                detail = self.map_navigator.describe_location(location_query)
                if detail:
                    extra = f" {detail}."
            return f"Here's the map to {location_query}.{extra}"

        if self.map_navigator and self.map_navigator.available:
            detail = self.map_navigator.describe_location(location_query)
            if detail:
                return f"I don't have a map image, but I know this spot: {detail}."
        return f"Sorry, I don't have a map for '{location_query}'."

    async def get_campus_directions(
        self, start_location: str, destination: str, context: RunContext
    ) -> str:
        if not self.map_navigator or not self.map_navigator.available:
            return (
                "I don't have the campus navigation graph loaded yet, "
                "but I can show you a map image if you tell me the place name."
            )
        route = self.map_navigator.directions(start_location, destination)
        if not route:
            return (
                f"I couldn't find a route from {start_location} to {destination}. "
                f"Known places: {', '.join(self.map_navigator.list_locations())}."
            )
        return f"Go this way: {route}."
