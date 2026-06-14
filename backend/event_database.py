from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from event_indexer import POSTER_CATEGORIES, VALID_EXTENSIONS, index_posters


class EventDatabase:
    def __init__(self, persist_directory: Path):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="campus_events",
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )

    def add_events(self, events: list[dict]) -> None:
        if not events:
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for i, event in enumerate(events):
            text_desc = (
                f"{event.get('title', '')} on {event.get('date', '')} "
                f"at {event.get('time', '')}. {event.get('description', '')}"
            )
            ids.append(f"event_{i}_{event.get('source_file', 'unknown')}")
            documents.append(text_desc)
            metadatas.append({k: str(v) for k, v in event.items()})

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Added {len(ids)} items to event database")

    def query_events(self, query_text: str, n_results: int = 3) -> list[dict]:
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        formatted: list[dict] = []
        if results["metadatas"] and results["metadatas"][0]:
            formatted.extend(results["metadatas"][0])
        return formatted

    def has_data(self) -> bool:
        return self.collection.count() > 0


def _compute_events_manifest(assets_dir: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for category in POSTER_CATEGORIES:
        cat_dir = assets_dir / category
        if not cat_dir.exists():
            continue
        for file_path in sorted(cat_dir.iterdir()):
            if file_path.suffix.lower() in VALID_EXTENSIONS:
                manifest[f"{category}/{file_path.name}"] = hashlib.md5(
                    file_path.read_bytes()
                ).hexdigest()
    return manifest


def build_event_database(assets_dir: Path) -> EventDatabase:
    db_path = Path(__file__).parent / "event_db"
    db_path.mkdir(exist_ok=True)
    manifest_path = db_path / "event_manifest.json"
    extracted_path = db_path / "extracted_events.json"

    db = EventDatabase(db_path)
    current_manifest = _compute_events_manifest(assets_dir)

    saved_manifest: dict[str, str] = {}
    if manifest_path.exists():
        try:
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            saved_manifest = {}

    if current_manifest == saved_manifest and db.has_data():
        print(
            f"Event DB up-to-date ({len(current_manifest)} poster(s), skipping re-index)"
        )
        return db

    changed = set(current_manifest) ^ set(saved_manifest)
    print(f"Posters changed ({len(changed)} file(s) differ). Re-indexing...")
    events = index_posters(assets_dir)
    db.add_events(events)

    extracted_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved ({len(current_manifest)} file(s) tracked)")
    return db
