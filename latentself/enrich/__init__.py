import json
from pathlib import Path
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError
from subprocess import SubprocessError

from . import docs, images
from ..backends import InferenceUnavailable
from ..ingest.gemini import _rebuild_fts


def enrich(con, llm, *, do_images=True, do_docs=True, limit=None, retry_errors=False, emit=print, source=None):
    stats = dict(success=0, skipped=0, failed=0, limited=0, failures=[], omitted_chars=0, omitted_frames=0)
    rows = con.execute("""SELECT a.*, t.prompt FROM attachment a JOIN turn t ON t.id=a.turn_id
                          WHERE (? IS NULL OR t.source=?) ORDER BY t.ts,a.pos,a.id""",(source, source)).fetchall()
    for row in rows:
        name = row["stored_name"]
        suffix = Path(name).suffix.lower()
        document = suffix in (".pdf", ".docx")
        archive = suffix == ".zip"
        visual = archive or (row["mime"] or "").startswith("image/") or suffix in (".heic", ".heif")
        if not ((do_docs and document) or (do_images and visual)):
            stats["skipped"] += 1
            continue
        done = row["ocr"] is not None if document else row["caption"] is not None
        if done or not row["path"] or (row["enrich_error"] and not retry_errors):
            stats["skipped"] += 1
            continue
        if limit is not None and stats["success"] + stats["failed"] >= limit:
            stats["limited"] += 1
            continue
        for attempt in range(2):
            try:
                path = Path(row["path"])
                origin = row["origin"]
                if document:
                    ocr, omitted = docs.extract(path, name)
                    value = dict(caption=None, ocr=ocr, tags=[], content_kind="document")
                    model = "pdftotext" if suffix == ".pdf" else "docx-xml"
                    frames = 0
                else:
                    data, frames = images.prepare(path, archive)
                    if not data:
                        stats["skipped"] += 1
                        break
                    if archive:
                        origin = "live_frame"
                    value = images.describe(llm, row["prompt"], data, origin == "generated")
                    omitted = max(0, len(row["prompt"]) - 300)
                    model = llm.model
                with con:
                    con.execute("""UPDATE attachment SET caption=?,ocr=?,tags=?,content_kind=?,
                        origin=?,enrich_model=?,enrich_at=CURRENT_TIMESTAMP,enrich_error=NULL WHERE id=?""",
                        (value["caption"], value["ocr"], json.dumps(value["tags"], ensure_ascii=False),
                         value["content_kind"], origin, model, row["id"]))
                    _rebuild_fts(con, row["turn_id"])
                stats["success"] += 1
                stats["omitted_chars"] += omitted
                stats["omitted_frames"] += frames
                emit(f"success {stats['success']} {row['id']}")
                break
            except InferenceUnavailable as exc:
                if attempt == 0:
                    continue
                stats["interrupted"] = str(exc)
                return stats
            except (OSError, BadZipFile, ParseError, ValueError, TypeError, KeyError, SubprocessError) as exc:
                if attempt == 0:
                    continue
                error = str(exc)
                with con:
                    con.execute("UPDATE attachment SET enrich_error=? WHERE id=?", (error, row["id"]))
                stats["failed"] += 1
                stats["failures"].append({"id": row["id"], "error": error})
    return stats
