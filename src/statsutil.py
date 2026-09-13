"""Tiny merge-and-save helper so each pipeline stage can drop its own
summary numbers in one place for the dashboard to read, without any stage
needing to know about the others."""
import json
from pathlib import Path

STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "stats.json"


def update_stats(section: str, values: dict) -> None:
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
    data[section] = values
    STATS_PATH.write_text(json.dumps(data, indent=2))


def load_stats() -> dict:
    return json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
