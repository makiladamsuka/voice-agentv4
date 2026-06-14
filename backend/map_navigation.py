"""Campus map graph loader — optional JSON floor plan for location lookup."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


class MapNavigator:
    def __init__(self, graph_path: Path | None = None):
        self.graph_path = graph_path or Path(__file__).parent / "data" / "map_graph_floor_1.json"
        self._graph: dict[str, Any] = {}
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.graph_path.is_file():
            print(f"Map graph not found at {self.graph_path} (static map images still work)")
            return
        try:
            raw = json.loads(self.graph_path.read_text(encoding="utf-8"))
            self._graph = raw
            nodes = raw.get("nodes") or raw.get("locations") or []
            if isinstance(nodes, dict):
                for key, node in nodes.items():
                    self._nodes[str(key).lower()] = node
            elif isinstance(nodes, list):
                for node in nodes:
                    node_id = str(node.get("id") or node.get("name") or "")
                    if node_id:
                        self._nodes[node_id.lower()] = node
                    label = str(node.get("name") or node.get("label") or "")
                    if label:
                        self._nodes[label.lower()] = node
            edges = raw.get("edges") or raw.get("connections") or []
            self._edges = edges if isinstance(edges, list) else []
            print(f"Map graph loaded: {len(self._nodes)} location(s)")
        except Exception as exc:
            print(f"Failed to load map graph: {exc}")

    @property
    def available(self) -> bool:
        return bool(self._nodes)

    def list_locations(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for node in self._nodes.values():
            label = str(node.get("name") or node.get("label") or node.get("id") or "")
            if label and label not in seen:
                seen.add(label)
                names.append(label)
        return sorted(names)

    def _resolve_node(self, query: str) -> str | None:
        q = query.lower().strip()
        if q in self._nodes:
            return q
        for key, node in self._nodes.items():
            label = str(node.get("name") or node.get("label") or key).lower()
            if q in label or label in q:
                return key
        return None

    def describe_location(self, query: str) -> str | None:
        node_key = self._resolve_node(query)
        if not node_key:
            return None
        node = self._nodes[node_key]
        parts = [str(node.get("name") or node.get("label") or node_key)]
        for field in ("floor", "building", "description", "room"):
            value = node.get(field)
            if value:
                parts.append(f"{field}: {value}")
        return ", ".join(parts)

    def _node_id(self, node_key: str) -> str:
        node = self._nodes.get(node_key, {})
        return str(node.get("id") or node_key).lower()

    def directions(self, start: str, end: str) -> str | None:
        start_key = self._resolve_node(start)
        end_key = self._resolve_node(end)
        if not start_key or not end_key:
            return None
        start_id = self._node_id(start_key)
        end_id = self._node_id(end_key)
        if start_id == end_id:
            return "You are already at that location."

        adjacency: dict[str, list[str]] = {}
        for edge in self._edges:
            a = str(edge.get("from") or edge.get("a") or "").lower()
            b = str(edge.get("to") or edge.get("b") or "").lower()
            if not a or not b:
                continue
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        queue: deque[str] = deque([start_id])
        parent: dict[str, str | None] = {start_id: None}
        while queue:
            current = queue.popleft()
            if current == end_id:
                break
            for neighbor in adjacency.get(current, []):
                if neighbor not in parent:
                    parent[neighbor] = current
                    queue.append(neighbor)

        if end_id not in parent:
            return None

        id_to_label: dict[str, str] = {}
        for node in self._nodes.values():
            node_id = str(node.get("id") or "").lower()
            if node_id:
                id_to_label[node_id] = str(node.get("name") or node.get("label") or node_id)

        path: list[str] = []
        cursor: str | None = end_id
        while cursor is not None:
            path.append(id_to_label.get(cursor, cursor))
            cursor = parent[cursor]
        path.reverse()
        return " -> ".join(path)
