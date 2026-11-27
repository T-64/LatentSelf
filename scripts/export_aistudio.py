"""Export owned Google AI Studio conversations and their referenced media to a new private snapshot."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from urllib.parse import urlsplit

PROMPT = "application/vnd.google-makersuite.prompt"


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)


def drive(args, *, allow_missing=False):
    for delay in (0, 5, 20, 60, 90):
        if delay:
            time.sleep(delay)
        run = subprocess.run(["rclone", *args, "--contimeout", "15s", "--timeout", "90s",
                              "--low-level-retries", "3"], capture_output=True)
        if not run.returncode:
            return run.stdout
        error = run.stderr.lower()
        if allow_missing and b"404" in error and (b"not found" in error or b"notfound" in error):
            return None
        if b"403" in error and b"quota" not in error and b"rate" not in error:
            if allow_missing and (b"insufficientfilepermissions" in error or b"access denied" in error):
                return None
            break
    raise RuntimeError("Drive read failed; check rclone authentication, connectivity and quota")


def normalize(raw, record, media_reference):
    prompt = raw["chunkedPrompt"]
    messages = []
    counts = Counter()
    for index, chunk in enumerate(prompt.get("chunks", [])):
        if chunk.get("isThought"):
            counts["thought_chunks_excluded"] += 1
            continue
        parts = chunk.get("parts") or []
        visible = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        text = visible if any(part.get("thought") for part in parts) else chunk.get("text") or visible
        msg = dict(index=index, role=chunk["role"], time=chunk.get("createTime"), text=text, attachments=[])
        for key, value in chunk.items():
            if isinstance(value, dict) and key.startswith("drive") and "id" in value:
                msg["attachments"].append(media_reference(key, value))
            elif isinstance(value, dict) and key.startswith("inline") and "data" in value:
                msg["attachments"].append(media_reference(key, value))
            elif key == "youtubeVideo":
                msg["attachments"].append(dict(kind=key, reference=value))
                counts["external_video_refs"] += 1
            elif key in ("branchParent", "branchChildren", "errorMessage", "finishReason"):
                msg[key] = value
        counts["messages_" + msg["role"]] += 1
        messages.append(msg)
    counts["unsent_nonempty_drafts_excluded"] += sum(
        bool(c.get("text") or any(k.startswith(("inline", "drive")) for k in c))
        for c in prompt.get("pendingInputs", []))
    entry = dict(source="aistudio", drive_id=record["id"], title=record["name"], url=record["webViewLink"],
                 drive_created=record["createdTime"], drive_modified=record["modifiedTime"],
                 source_sha256=record["sha256Checksum"], model=raw.get("runSettings", {}).get("model"),
                 messages=messages)
    return entry, counts


def export(remote, folder, output, reuse=()):
    if not re.fullmatch(r"[\w.-]+:", remote):
        raise ValueError("expected an rclone remote such as remote:")
    output = Path(output).expanduser()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    objects = output / "objects"
    objects.mkdir()
    listed = json.loads(drive(["lsjson", remote + folder, "--recursive", "--hash", "--files-only"]))
    owned = json.loads(drive(["backend", "query", remote,
                            f"trashed = false and 'me' in owners and mimeType = '{PROMPT}'"]))
    write_json(output / "drive-inventory.json", listed)
    write_json(output / "owned-prompts.json", owned)
    candidates = {r["ID"]: r for r in listed}
    for row in owned:
        if urlsplit(row["webViewLink"]).netloc != "aistudio.google.com":
            raise ValueError("unexpected prompt source URL")
        prev = candidates.get(row["id"])
        if prev and prev["Hashes"]["sha256"] != row["sha256Checksum"]:
            raise ValueError("a prompt changed during inventory; use a new snapshot")
        candidates[row["id"]] = dict(ID=row["id"], Name=row["name"], Size=int(row["size"]),
                                     MimeType=PROMPT, Hashes={"sha256":row["sha256Checksum"]})
    cache = {}
    for root in reuse:
        for path in Path(root).expanduser().iterdir():
            if path.is_file():
                cache[checksum(path)] = path

    def materialize(meta):
        h = meta["Hashes"]["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", h):
            raise ValueError("missing Drive checksum")
        path = objects / h
        if path.exists():
            if checksum(path) != h:
                raise ValueError("invalid local object")
            return
        if h in cache:
            shutil.copyfile(cache[h], path)
        else:
            drive(["backend", "copyid", remote, meta["ID"], str(path.resolve()), "--immutable"])
        if path.stat().st_size != meta["Size"] or checksum(path) != h:
            raise ValueError("object changed during export")

    def fetch_many(rows):
        unique = {m["Hashes"]["sha256"]:m for m in rows}
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(materialize, unique.values()))

    fetch_many(candidates[r["id"]] for r in owned)
    refs = set()
    media = {}
    def media_reference(key, value):
        if key.startswith("drive"):
            refs.add(value["id"])
            return dict(kind=key, key=value["id"])
        data = base64.b64decode(value["data"], validate=True)
        h = sha256(data).hexdigest()
        path = objects / h
        if not path.exists():
            with path.open("xb") as f:
                f.write(data)
        media["inline:" + h] = dict(path=str(path.relative_to(output)), sha256=h, size=len(data),
                                    name=None, mime=value.get("mimeType"))
        return dict(kind=key, key="inline:" + h)

    counts = Counter()
    with (output / "conversations.jsonl").open("x", encoding="utf-8") as corpus:
        for r in sorted(owned, key=lambda r:r["id"]):
            raw = json.loads((objects / r["sha256Checksum"]).read_bytes())
            row, stats = normalize(raw, r, media_reference)
            corpus.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts.update(stats)
            counts["conversations"] += 1
    fetch_many(candidates[i] for i in refs if i in candidates)
    for i in sorted(refs):
        if i in candidates:
            meta = candidates[i]
            h = meta["Hashes"]["sha256"]
            media[i] = dict(path="objects/"+h, sha256=h, size=meta["Size"], name=meta["Name"], mime=meta["MimeType"])
        else:
            with TemporaryDirectory(dir=output) as tmp:
                payload = drive(["backend","copyid",remote,i,str(Path(tmp).resolve())+"/","--immutable"],allow_missing=True)
                if payload is None:
                    media[i] = dict(path=None, reason="unavailable")
                    continue
                (path,) = Path(tmp).iterdir()
                h = checksum(path)
                target = objects / h
                if not target.exists():
                    shutil.copyfile(path,target)
                media[i] = dict(path="objects/"+h, sha256=h, size=path.stat().st_size,
                                name=path.name, mime=mimetypes.guess_type(path.name)[0])
    write_json(output / "media.json", media)
    counts["resolved_media_objects"] = sum(bool(m["path"]) for m in media.values())
    counts["unavailable_media_objects"] = len(media) - counts["resolved_media_objects"]
    write_json(output / "corpus-summary.json", counts)
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="configured rclone Drive remote ending in a colon")
    parser.add_argument("--folder", default="Google AI Studio")
    parser.add_argument("--output", required=True, help="new private directory")
    parser.add_argument("--reuse", action="append", default=[], help="directory of previously downloaded objects")
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(export(args.remote,args.folder,args.output,args.reuse),ensure_ascii=False))
