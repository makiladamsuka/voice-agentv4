"""Fuzzy matching for event posters and location maps."""

from __future__ import annotations

import base64
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional


class ImageManager:
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.events_dir = assets_dir / "events"
        self.maps_dir = assets_dir / "maps"
        self.fallback_dir = assets_dir / "fallback"

        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

        print("MediaManager initialized")
        print(f"   Events: {self.events_dir}")
        print(f"   Maps: {self.maps_dir}")

    def _fuzzy_match(
        self, query: str, candidates: List[str], threshold: float = 0.5
    ) -> Optional[str]:
        query_lower = query.lower().strip()
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            candidate_name = Path(candidate).stem.lower()
            candidate_clean = candidate_name.replace("_", " ").replace("-", " ")
            score = SequenceMatcher(None, query_lower, candidate_clean).ratio()
            if query_lower in candidate_clean:
                score = max(score, 0.8)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= threshold:
            return best_match
        return None

    def _get_all_images(self, directory: Path) -> List[str]:
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        images = []
        if directory.exists():
            for file in directory.iterdir():
                if file.suffix.lower() in image_extensions:
                    images.append(file.name)
        return images

    def find_event_image(self, query: str) -> Optional[Path]:
        available_events = self._get_all_images(self.events_dir)
        if not available_events:
            return None
        matched_file = self._fuzzy_match(query, available_events)
        if matched_file:
            return self.events_dir / matched_file
        return None

    def list_available_events(self) -> List[str]:
        available_events = self._get_all_images(self.events_dir)
        event_names = []
        for filename in available_events:
            name = Path(filename).stem.replace("-", " ").replace("_", " ")
            name = " ".join(word.capitalize() for word in name.split())
            event_names.append(name)
        return event_names

    def find_location_map(self, query: str) -> Optional[Path]:
        available_maps = self._get_all_images(self.maps_dir)
        if not available_maps:
            return None
        matched_file = self._fuzzy_match(query, available_maps)
        if matched_file:
            return self.maps_dir / matched_file
        return None

    def get_fallback_image(self) -> Optional[Path]:
        fallback_files = self._get_all_images(self.fallback_dir)
        if fallback_files:
            return self.fallback_dir / fallback_files[0]
        return None

    def encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
