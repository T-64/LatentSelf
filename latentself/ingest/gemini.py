"""Gemini Takeout (My Activity / Gemini Apps / MyActivity.json) -> unified model.

Field mapping is documented in docs/DESIGN.md §2.2. Every activity is kept; `kind`
separates dialogue from feedback, canvases and housekeeping events.
"""
import hashlib
import json
import mimetypes
import re
import sqlite3
from bisect import bisect_right
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

SOURCE = "gemini"
DIALOGUE = ("prompt", "branch", "link_import", "live")
LINK_WINDOW_S = 30 * 60  # unlinked canvas/feedback attach to a dialogue turn this close before it

# Order matters: longer prefixes first.
PREFIXES = [
    ("Created Gemini Canvas titled", "canvas"),
    ("Added chat from link:", "link_import"),
    ("Cleared previous feedback", "cleared"),
    ("Selected preferred draft", "draft_select"),
    ("Answered survey question", "survey"),
    ("Gave feedback:", "feedback"),
    ("Used Gemini Apps", "used"),
    ("Live Prompt", "live"),
    ("Prompted", "prompt"),
    ("Branched", "branch"),
]

CONV_RE = re.compile(r"gemini\.google\.com/app/([0-9a-f]+)")
DUP_SUFFIX_RE = re.compile(r"^(?P<stem>.+?)\((?P<n>\d+)\)(?P<ext>\.[^.]*)?$")


def classify(title: str):
    for prefix, kind in PREFIXES:
        if title.startswith(prefix):
            return kind, title[len(prefix):].strip()
    return "other", title.strip()


def turn_id(ts: str, title: str) -> str:
    return hashlib.sha1(f"{SOURCE}|{ts}|{title}".encode()).hexdigest()[:16]


def parse_ts(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


class _Text(HTMLParser):
    """HTML -> readable plain text that keeps list/heading/table structure."""
    BLOCK = {"p", "div", "section", "blockquote", "pre", "table", "thead", "tbody", "ul", "ol", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK or tag in ("h1", "h2", "h3", "h4", "h5", "tr", "br"):
            self.out.append("\n")
        if tag == "li":
            self.out.append("\n- ")
        elif tag in ("h1", "h2", "h3", "h4", "h5"):
            self.out.append("#" * int(tag[1]) + " ")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.BLOCK or tag in ("h1", "h2", "h3", "h4", "h5"):
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    text = "".join(p.out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class AttachmentResolver:
    """Takeout stores files flat; names sometimes lose their extension or gain (n) on collision."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.names = set(p.name for p in folder.iterdir()) if folder.exists() else set()
        self.used_dups = set()

    def resolve(self, stored: str):
        if stored in self.names:
            return self.folder / stored, "exact"
        stem, dot, ext = stored.rpartition(".")
        if dot and stem in self.names:
            return self.folder / stem, "no_ext"
        base = stem if dot else stored
        suffix = f".{ext}" if dot else ""
        n = 1
        while f"{base}({n}){suffix}" in self.names:
            cand = f"{base}({n}){suffix}"
            if cand not in self.used_dups:
                self.used_dups.add(cand)
                return self.folder / cand, "dup_suffix"
            n += 1
        return None, "missing"


def _orig_names(entry):
    """Map stored name -> name the user uploaded, from subtitles ("-  name", url=stored)."""
    out = {}
    for s in entry.get("subtitles") or []:
        if "url" in s and s.get("name", "").startswith("-"):
            out[s["url"]] = s["name"].lstrip("- ").strip()
    return out


def load(snapshot: Path):
    with open(snapshot / "MyActivity.json", encoding="utf-8") as f:
        return json.load(f)


def ingest(con: sqlite3.Connection, snapshot: Path) -> dict:
    entries = load(snapshot)
    resolver = AttachmentResolver(snapshot / "attachments")
    snap = snapshot.name
    stats = {"entries": len(entries)}

    rows, convs, answers, atts, canvases, signals = [], [], [], [], [], []
    # Process oldest first so (n) duplicate suffixes are assigned in upload order.
    for e in sorted(entries, key=lambda x: x["time"]):
        kind, prompt = classify(e["title"])
        tid = turn_id(e["time"], e["title"])
        rows.append((tid, SOURCE, kind, e["time"], prompt, snap))
        for d in e.get("details") or []:
            m = CONV_RE.search(d.get("url", ""))
            if m:
                convs.append((tid, m.group(1), 0))
        for i, h in enumerate(e.get("safeHtmlItem") or []):
            answers.append((tid, i, h["html"], html_to_text(h["html"])))
        stored = list(e.get("attachedFiles") or [])
        if e.get("imageFile") and e["imageFile"] not in stored:
            stored.append(e["imageFile"])
        orig = _orig_names(e)
        for pos, name in enumerate(stored):
            path, match = resolver.resolve(name)
            mime = mimetypes.guess_type(name)[0]
            origin = "user"
            if name.startswith("watermarked_img_"):
                origin = "generated"
            elif kind == "live" and name.lower().endswith(".zip"):
                origin = "live_frame"
            atts.append((f"{tid}:{pos}", tid, pos, orig.get(name), name, str(path) if path else None, match, mime, origin))
        if kind == "canvas":
            subs = e.get("subtitles") or []
            content = subs[0]["name"] if subs else None
            canvases.append((tid, prompt or None, content))
        elif kind in ("feedback", "draft_select"):
            signals.append((tid, kind, prompt or None))

    with con:
        con.executemany("""INSERT INTO turn (id,source,kind,ts,prompt,snapshot) VALUES (?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET snapshot=excluded.snapshot, kind=excluded.kind,
            ts=excluded.ts, prompt=excluded.prompt""", rows)
        ids = [(r[0],) for r in rows]
        for table in ("turn_conv", "answer", "canvas", "signal"):
            con.executemany(f"DELETE FROM {table} WHERE turn_id = ?", ids)
        con.executemany("INSERT OR IGNORE INTO turn_conv VALUES (?,?,?)", convs)
        con.executemany("INSERT INTO answer VALUES (?,?,?,?)", answers)
        _merge_attachments(con, ids, atts)
        con.executemany("INSERT INTO canvas VALUES (?,?,?)", canvases)
        con.executemany("INSERT INTO signal (turn_id, kind, value) VALUES (?,?,?)", signals)
        _link_by_time(con)
        _rebuild_conversations(con)
        _rebuild_fts(con)
        con.execute("INSERT OR REPLACE INTO meta VALUES ('gemini_snapshot', ?)", (snap,))

    stats.update({
        "turns": len(rows),
        "kinds": dict(con.execute("SELECT kind, COUNT(*) FROM turn GROUP BY kind ORDER BY 2 DESC").fetchall()),
        "conversations": con.execute("SELECT COUNT(*) FROM conversation").fetchone()[0],
        "multi_conv_turns": con.execute("SELECT COUNT(*) FROM (SELECT turn_id FROM turn_conv WHERE inferred=0 GROUP BY turn_id HAVING COUNT(*)>1)").fetchone()[0],
        "attachments": dict(con.execute("SELECT match, COUNT(*) FROM attachment GROUP BY match").fetchall()),
        "canvas_inferred_conv": con.execute("SELECT COUNT(*) FROM turn_conv tc JOIN turn t ON t.id=tc.turn_id WHERE tc.inferred=1 AND t.kind='canvas'").fetchone()[0],
        "signals_linked": con.execute("SELECT COUNT(*) FROM signal WHERE target_id IS NOT NULL").fetchone()[0],
    })
    return stats


def _merge_attachments(con, turn_ids, attachments):
    # Positions identify attachments within a turn. A new file at the same
    # position must not inherit the previous file's caption or image vector.
    con.execute("CREATE TEMP TABLE incoming_file (id TEXT PRIMARY KEY, stored_name TEXT)")
    con.executemany("INSERT OR IGNORE INTO incoming_file VALUES (?,?)", ((a[0], a[4]) for a in attachments))
    con.executemany("""DELETE FROM attachment WHERE turn_id=? AND NOT EXISTS (
        SELECT 1 FROM incoming_file i WHERE i.id=attachment.id AND i.stored_name=attachment.stored_name
    )""", turn_ids)
    con.execute("""DELETE FROM embedding WHERE item_kind='attachment'
        AND NOT EXISTS (SELECT 1 FROM attachment WHERE id=embedding.item_id)""")
    con.execute("""DELETE FROM embedding WHERE item_kind IN ('turn','turn_self')
        AND NOT EXISTS (SELECT 1 FROM turn WHERE id=embedding.item_id)""")
    con.executemany("""INSERT INTO attachment
        (id,turn_id,pos,orig_name,stored_name,path,match,mime,origin) VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET orig_name=excluded.orig_name, path=excluded.path,
        match=excluded.match, mime=excluded.mime, origin=excluded.origin""", attachments)
    con.execute("DROP TABLE incoming_file")


def _link_by_time(con):
    """Canvases without a URL inherit the conversation of the dialogue turn just before them;
    feedback/draft signals point at that turn."""
    dlg = con.execute(
        f"SELECT t.id, t.ts FROM turn t WHERE t.source=? AND t.kind IN ({','.join('?' * len(DIALOGUE))}) ORDER BY t.ts",
        (SOURCE, *DIALOGUE)).fetchall()
    times = [parse_ts(r["ts"]) for r in dlg]

    def nearest_before(ts):
        t = parse_ts(ts)
        i = bisect_right(times, t) - 1
        if i >= 0 and t - times[i] <= LINK_WINDOW_S:
            return dlg[i]["id"]
        return None

    for r in con.execute("SELECT t.id, t.ts FROM turn t WHERE t.kind='canvas' AND NOT EXISTS (SELECT 1 FROM turn_conv c WHERE c.turn_id=t.id)").fetchall():
        target = nearest_before(r["ts"])
        if target:
            con.execute("INSERT OR IGNORE INTO turn_conv SELECT ?, conv_id, 1 FROM turn_conv WHERE turn_id=?", (r["id"], target))
    for r in con.execute("SELECT s.turn_id, t.ts FROM signal s JOIN turn t ON t.id=s.turn_id").fetchall():
        con.execute("UPDATE signal SET target_id=? WHERE turn_id=?", (nearest_before(r["ts"]), r["turn_id"]))


def _rebuild_conversations(con):
    con.execute("""DELETE FROM conversation WHERE source=? AND NOT EXISTS (
        SELECT 1 FROM turn_conv WHERE conv_id=conversation.id)""", (SOURCE,))
    con.execute("""
        INSERT INTO conversation (id, source, first_ts, last_ts, n_turns)
        SELECT tc.conv_id, ?, MIN(t.ts), MAX(t.ts), SUM(t.kind IN ('prompt','branch','link_import','live'))
        FROM turn_conv tc JOIN turn t ON t.id = tc.turn_id
        WHERE t.source = ?
        GROUP BY tc.conv_id
        ON CONFLICT(id) DO UPDATE SET first_ts=excluded.first_ts, last_ts=excluded.last_ts,
        n_turns=excluded.n_turns""", (SOURCE, SOURCE))


def _rebuild_fts(con):
    con.execute("DELETE FROM fts_turn")
    con.execute("""
        INSERT INTO fts_turn (turn_id, prompt, answer, extra)
        SELECT t.id, t.prompt,
               COALESCE((SELECT a.text FROM answer a WHERE a.turn_id=t.id AND a.draft_idx=0), ''),
               COALESCE((SELECT group_concat(COALESCE(x.orig_name,'') || ' ' || COALESCE(x.caption,'') || ' ' || COALESCE(x.ocr,'') || ' ' || COALESCE(x.transcript,''), ' ')
                         FROM attachment x WHERE x.turn_id=t.id), '')
               || COALESCE((SELECT ' ' || c.title || ' ' || c.content FROM canvas c WHERE c.turn_id=t.id), '')
        FROM turn t
        WHERE t.kind IN ('prompt','branch','link_import','live','canvas')""")
