import re
from datetime import datetime
from pathlib import Path


def slugify_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "unknown"


def generate_run_id(target_name: str, runs_root: Path = Path("data/runs")) -> str:
    """
    Generate YYYY-MM-DD_HHMM_target_name_slug.
    If already taken, append _002, _003, etc.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = f"{timestamp}_{slugify_name(target_name)}"

    candidate = base
    counter = 2

    while (runs_root / candidate).exists():
        candidate = f"{base}_{counter:03d}"
        counter += 1

    return candidate