"""Paths and endpoints. Code never stores data; everything lives under $LATENTSELF_DATA."""
import os
import re
from pathlib import Path

DATA = Path(os.environ.get("LATENTSELF_DATA", "~/data")).expanduser()
OUT = Path(os.environ.get("LATENTSELF_OUT", "~/t/_LatentSelf")).expanduser()
DB_PATH = Path(os.environ.get("LATENTSELF_DB", DATA / "processed" / "latentself" / "ls.db")).expanduser()

GEMINI_RAW = DATA / "raw" / "gemini"

_DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def latest_gemini_snapshot(root: Path = GEMINI_RAW) -> Path:
    """Newest raw/gemini/<YYYY-MM-DD>/ that contains MyActivity.json."""
    snaps = sorted(p for p in root.iterdir() if p.is_dir() and _DATE_DIR.match(p.name) and (p / "MyActivity.json").exists())
    if not snaps:
        raise FileNotFoundError(f"no dated snapshot with MyActivity.json under {root}")
    return snaps[-1]
