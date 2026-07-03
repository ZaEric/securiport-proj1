import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def path_to_api_str(path: Path | str) -> str:
    """
    Convert file paths to API-friendly forward-slash strings.

    Prevents Windows paths from showing as data\\runs\\... in JSON responses.
    """
    return Path(path).as_posix()