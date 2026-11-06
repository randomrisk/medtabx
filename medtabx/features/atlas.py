"""Atlas metadata helpers for the OpenADS atlas pack."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from .openads_bridge import get_openads_atlas_dir


@lru_cache(maxsize=1)
def load_openads_feature_metadata() -> Dict[str, Dict[str, List[str]]]:
    """Load atlas label and QFV metadata from the OpenADS atlas pack."""
    names_path = Path(get_openads_atlas_dir()) / "names.json"
    return json.loads(names_path.read_text())
