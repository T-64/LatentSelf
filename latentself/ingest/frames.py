"""Recognize Gemini Live camera frames from a ZIP's directory."""
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile


def frame_names(archive):
    numbered = []
    for name in archive.namelist():
        match = re.fullmatch(r"frame_(\d+)\.png", Path(name).name, re.I)
        if match:
            numbered.append((int(match[1]),name))
    return [name for _,name in sorted(numbered)]


def is_frame_archive(path, fallback=False):
    if path is None:
        return fallback
    try:
        with ZipFile(path) as archive:
            return bool(frame_names(archive))
    except (BadZipFile,OSError):
        return fallback
