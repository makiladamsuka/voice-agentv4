from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from openai import OpenAI

POSTER_CATEGORIES = ("events", "competitions", "posts")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _vision_client() -> tuple[OpenAI, str] | tuple[None, None]:
    if os.getenv("OPENROUTER_API_KEY"):
        return (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            ),
            "google/gemini-2.5-flash",
        )
    if os.getenv("GROQ_API_KEY"):
        return (
            OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            "llama-3.3-70b-versatile",
        )
    return None, None


def index_posters(assets_dir: Path) -> list[dict]:
    """Scan events/competitions/posts posters and extract structured metadata."""
    client, model = _vision_client()
    if client is None:
        print("No OPENROUTER_API_KEY or GROQ_API_KEY — skipping poster indexing")
        return []

    extracted: list[dict] = []
    print(f"Scanning posters in {assets_dir} ({', '.join(POSTER_CATEGORIES)})...")

    for category in POSTER_CATEGORIES:
        cat_dir = assets_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for file_path in sorted(cat_dir.iterdir()):
            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            print(f"   Processing {category}/{file_path.name}...")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Extract details from this poster/image. Return JSON with keys: "
                                        "title, date, time, location, description. "
                                        "Do your best to extract any relevant information."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{encode_image(file_path)}",
                                    },
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=1000,
                )
                content = response.choices[0].message.content
                if not content:
                    continue
                event_data = json.loads(content)
                if not event_data:
                    continue
                event_data["source_file"] = file_path.name
                event_data["category"] = category
                extracted.append(event_data)
                print(f"   Extracted: {event_data.get('title', 'Unknown')} ({category})")
            except Exception as exc:
                print(f"   Failed to process {category}/{file_path.name}: {exc}")

    print(f"Poster indexing complete ({len(extracted)} item(s))")
    return extracted
