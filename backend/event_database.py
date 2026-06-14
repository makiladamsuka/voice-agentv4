from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from event_indexer import index_posters


class EventDatabase:
    def __init__(self, persist_directory):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name="campus_events",
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        )

    def add_events(self, events):
        if not events:
            return

        ids = []
        documents = []
        metadatas = []

        for i, event in enumerate(events):
            text_desc = (
                f"{event.get('title', '')} on {event.get('date', '')} "
                f"at {event.get('time', '')}. {event.get('description', '')}"
            )
            ids.append(f"event_{i}_{event.get('source_file', 'unknown')}")
            documents.append(text_desc)
            meta = {k: str(v) for k, v in event.items()}
            metadatas.append(meta)

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"Added {len(ids)} events to database")

    def query_events(self, query_text, n_results=3):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        formatted_events = []
        if results["metadatas"] and len(results["metadatas"]) > 0:
            for meta in results["metadatas"][0]:
                formatted_events.append(meta)
        return formatted_events

    def has_data(self) -> bool:
        return self.collection.count() > 0


def _compute_events_manifest(events_dir: Path) -> dict:
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    manifest = {}
    if events_dir.exists():
        for f in sorted(events_dir.iterdir()):
            if f.suffix.lower() in valid_extensions:
                manifest[f.name] = hashlib.md5(f.read_bytes()).hexdigest()
    return manifest


def build_event_database(assets_dir: Path):
    db_path = Path(__file__).parent / "event_db"
    db_path.mkdir(exist_ok=True)
    manifest_path = db_path / "event_manifest.json"

    db = EventDatabase(db_path)
    events_dir = assets_dir / "events"
    current_manifest = _compute_events_manifest(events_dir)

    saved_manifest = {}
    if manifest_path.exists():
        try:
            saved_manifest = json.loads(manifest_path.read_text())
        except Exception:
            saved_manifest = {}

    if current_manifest == saved_manifest and db.has_data():
        print(f"Event DB up-to-date ({len(current_manifest)} posters, skipping re-index)")
        return db

    changed = set(current_manifest) ^ set(saved_manifest)
    print(f"Events changed ({len(changed)} file(s) differ). Re-indexing...")
    events = index_posters(assets_dir)
    db.add_events(events)
    manifest_path.write_text(json.dumps(current_manifest, indent=2))
    print(f"Manifest saved ({len(current_manifest)} files tracked)")
    return db
