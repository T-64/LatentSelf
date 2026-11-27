"""Import an AI Studio snapshot while preserving speaker, chunk order and source links."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import file_digest, sha256
from html import escape
from itertools import batched
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .gemini import insert_fts


def groups(messages):
    """A submitted prompt can have several consecutive text and attachment chunks."""
    block = []
    for msg in messages:
        if msg["role"] not in ("user", "model"):
            raise ValueError("unsupported AI Studio speaker")
        if block and msg["role"] == "user" and block[-1]["role"] == "model":
            yield block
            block = []
        block.append(msg)
    if block:
        yield block


def event_time(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("AI Studio timestamp has no time zone")
    return date.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def create_source_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS aistudio_source(
        conv_id TEXT PRIMARY KEY REFERENCES conversation(id), file_id TEXT UNIQUE, url TEXT NOT NULL,
        sha256 TEXT NOT NULL, modified TEXT, created TEXT, model TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS aistudio_chunk(
        conv_id TEXT REFERENCES conversation(id), chunk_index INTEGER, role TEXT NOT NULL,
        turn_id TEXT REFERENCES turn(id), source_time TEXT, PRIMARY KEY(conv_id,chunk_index))""")
    con.execute("CREATE INDEX IF NOT EXISTS aistudio_chunk_turn ON aistudio_chunk(turn_id,chunk_index)")
    con.execute("""CREATE TABLE IF NOT EXISTS aistudio_reference(
        turn_id TEXT REFERENCES turn(id), chunk_index INTEGER, position INTEGER, kind TEXT,
        value TEXT, PRIMARY KEY(turn_id,chunk_index,position))""")


def load_media(folder):
    media = json.loads((folder / "media.json").read_bytes())
    verified = set()
    for obj in media.values():
        if not obj.get("path"):
            continue
        path = (folder / obj["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("media path lies outside the snapshot")
        if path not in verified:
            with path.open("rb") as stream:
                if file_digest(stream, "sha256").hexdigest() != obj["sha256"]:
                    raise ValueError("snapshot attachment checksum mismatch")
            verified.add(path)
        obj["path"] = str(path)
    return media


def insert_turn(con, cid, raw_hash, block, media, counts):
    users = [m for m in block if m["role"] == "user"]
    if not users:
        raise ValueError("cannot attribute an assistant-only message to the user")
    tid = sha256(f"{cid}/{users[0]['index']}".encode()).hexdigest()[:16]
    prompt = "\n\n".join(m["text"] for m in users if m["text"])
    event = next((event_time(m["time"]) for m in users if m.get("time")), "")
    con.execute("INSERT INTO turn(id,source,kind,ts,prompt,snapshot) VALUES(?,'aistudio','prompt',?,?,?)",
                (tid, event, prompt, raw_hash))
    con.execute("INSERT INTO turn_conv VALUES(?,?,0)", (tid, cid))
    reply = "\n\n".join(m["text"] for m in block if m["role"] == "model" and m["text"])
    if reply:
        con.execute("INSERT INTO answer VALUES(?,0,?,?)", (tid, escape(reply), reply))
    position = 0
    for msg in block:
        con.execute("INSERT INTO aistudio_chunk VALUES(?,?,?,?,?)",
                    (cid, msg["index"], msg["role"], tid, msg.get("time")))
        for j, ref in enumerate(msg["attachments"]):
            con.execute("INSERT INTO aistudio_reference VALUES(?,?,?,?,?)",
                        (tid, msg["index"], j, ref["kind"], json.dumps(ref, ensure_ascii=False)))
            if "key" not in ref:
                continue
            obj = media[ref["key"]]
            name, path = obj.get("name"), obj.get("path")
            con.execute("""INSERT INTO attachment(id,turn_id,pos,orig_name,stored_name,path,match,mime,origin)
                VALUES(?,?,?,?,?,?,?,?,?)""", (f"{tid}:{position}", tid, position, name, name or ref["key"],
                path, "exact" if path else "missing", obj.get("mime"), "user" if msg["role"] == "user" else "generated"))
            position += 1
    counts["turns"] += 1
    counts["undated_turns"] += not event
    return tid


def ingest(con, folder):
    """Import new sources atomically. Report revised files for explicit reconciliation."""
    folder = Path(folder).expanduser().resolve()
    media = load_media(folder)
    counts = Counter()
    inserted = []
    with con, (folder / "conversations.jsonl").open(encoding="utf-8") as corpus:
        con.execute("BEGIN IMMEDIATE")
        create_source_tables(con)
        for line in corpus:
            row = json.loads(line)
            file_id, raw_hash = row["drive_id"], row["source_sha256"]
            link = urlsplit(row["url"])
            if (row["source"] != "aistudio" or not re.fullmatch(r"[\w-]+", file_id, re.ASCII)
                    or link.scheme != "https" or link.netloc != "aistudio.google.com"
                    or not re.fullmatch(r"[a-f0-9]{64}", raw_hash)):
                raise ValueError("invalid AI Studio source metadata")
            if sha256((folder / "objects" / raw_hash).read_bytes()).hexdigest() != raw_hash:
                raise ValueError("AI Studio source checksum mismatch")
            cid = "aistudio:" + file_id
            old = con.execute("SELECT sha256 FROM aistudio_source WHERE conv_id=?", (cid,)).fetchone()
            if old:
                counts["skipped" if old[0] == raw_hash else "revised_pending"] += 1
                continue
            blocks = list(groups(row["messages"]))
            times = [event_time(m["time"]) for m in row["messages"] if m.get("time")]
            con.execute("""INSERT INTO conversation(id,source,first_ts,last_ts,n_turns,title)
                VALUES(?,'aistudio',?,?,?,?)""",
                        (cid, min(times, default=""), max(times, default=""), len(blocks), row["title"]))
            con.execute("INSERT INTO aistudio_source VALUES(?,?,?,?,?,?,?)",
                        (cid, file_id, row["url"], raw_hash, row["drive_modified"], row["drive_created"], row.get("model")))
            for block in blocks:
                inserted.append(insert_turn(con, cid, raw_hash, block, media, counts))
            counts["imported"] += 1
        # Each new turn has no FTS row yet. Existing sources and enrichments stay intact.
        for batch in batched(inserted, 300):
            insert_fts(con, batch)
    return dict(counts)
