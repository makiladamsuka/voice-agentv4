"""Fuzzy matching for event posters, competitions, posts, and campus maps."""

from __future__ import annotations

import base64
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

POSTER_CATEGORIES = ("events", "competitions", "posts")


class ImageManager:
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.events_dir = assets_dir / "events"
        self.competitions_dir = assets_dir / "competitions"
        self.posts_dir = assets_dir / "posts"
        self.maps_dir = assets_dir / "maps"
        self.fallback_dir = assets_dir / "fallback"
        self._extracted_index: dict[str, tuple[str, str]] = {}
        self._load_extracted_index()

        for folder in (
            self.events_dir,
            self.competitions_dir,
            self.posts_dir,
            self.maps_dir,
            self.fallback_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        print("MediaManager initialized")
        print(f"   Events: {self.events_dir}")
        print(f"   Competitions: {self.competitions_dir}")
        print(f"   Posts: {self.posts_dir}")
        print(f"   Maps: {self.maps_dir}")

    def _load_extracted_index(self) -> None:
        extracted_path = Path(__file__).parent / "event_db" / "extracted_events.json"
        if not extracted_path.is_file():
            return
        try:
            import json

            items = json.loads(extracted_path.read_text(encoding="utf-8"))
            for item in items:
                title = str(item.get("title") or "").strip().lower()
                source = str(item.get("source_file") or "")
                category = str(item.get("category") or "events")
                if title and source:
                    self._extracted_index[title] = (category, source)
        except Exception:
            self._extracted_index = {}

    def _category_dir(self, category: str) -> Path:
        return {
            "events": self.events_dir,
            "competitions": self.competitions_dir,
            "posts": self.posts_dir,
            "maps": self.maps_dir,
        }.get(category, self.events_dir)

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
        images: list[str] = []
        if directory.exists():
            for file in directory.iterdir():
                if file.suffix.lower() in image_extensions:
                    images.append(file.name)
        return images

    def _human_name(self, filename: str) -> str:
        name = Path(filename).stem.replace("-", " ").replace("_", " ")
        return " ".join(word.capitalize() for word in name.split())

    def find_poster(
        self, query: str, category: str | None = None
    ) -> Tuple[Optional[Path], Optional[str]]:
        query_lower = query.lower().strip()
        for title, (cat, source) in self._extracted_index.items():
            if category and cat != category:
                continue
            if query_lower in title or title in query_lower:
                path = self._category_dir(cat) / source
                if path.is_file():
                    return path, cat

        categories = (category,) if category else POSTER_CATEGORIES
        best: tuple[Optional[Path], Optional[str], float] = (None, None, 0.0)

        for cat in categories:
            directory = self._category_dir(cat)
            candidates = self._get_all_images(directory)
            if not candidates:
                continue
            matched = self._fuzzy_match(query, candidates)
            if not matched:
                continue
            candidate_name = Path(matched).stem.lower().replace("_", " ").replace("-", " ")
            query_lower = query.lower().strip()
            score = SequenceMatcher(None, query_lower, candidate_name).ratio()
            if query_lower in candidate_name:
                score = max(score, 0.8)
            if score > best[2]:
                best = (directory / matched, cat, score)

        return best[0], best[1]

    def find_event_image(self, query: str) -> Optional[Path]:
        path, _ = self.find_poster(query)
        return path

    def find_competition_image(self, query: str) -> Optional[Path]:
        path, _ = self.find_poster(query, category="competitions")
        return path

    def find_post_image(self, query: str) -> Optional[Path]:
        path, _ = self.find_poster(query, category="posts")
        return path

    def list_by_category(self, category: str) -> List[str]:
        directory = self._category_dir(category)
        names: list[str] = []
        for filename in self._get_all_images(directory):
            title = next(
                (
                    item.get("title")
                    for item in self._extracted_items()
                    if item.get("source_file") == filename
                    and item.get("category") == category
                    and item.get("title")
                ),
                None,
            )
            names.append(title if title else self._human_name(filename))
        return names

    def _extracted_items(self) -> list[dict]:
        extracted_path = Path(__file__).parent / "event_db" / "extracted_events.json"
        if not extracted_path.is_file():
            return []
        try:
            import json

            return json.loads(extracted_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def list_available_events(self) -> List[str]:
        names: list[str] = []
        for category in POSTER_CATEGORIES:
            for filename in self._get_all_images(self._category_dir(category)):
                names.append(self._human_name(filename))
        return names

    def list_campus_news(self) -> dict[str, list[str]]:
        return {category: self.list_by_category(category) for category in POSTER_CATEGORIES}

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
