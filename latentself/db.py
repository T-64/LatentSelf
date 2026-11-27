"""SQLite schema. One file holds the unified model, full-text index and later vectors/features."""
import sqlite3
from pathlib import Path

from .ingest.frames import is_frame_archive

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Every Takeout activity becomes one row; `kind` says what it is.
CREATE TABLE IF NOT EXISTS turn (
    id        TEXT PRIMARY KEY,          -- stable hash of source|time|title
    source    TEXT NOT NULL,             -- gemini | aistudio | claude
    kind      TEXT NOT NULL,             -- prompt | branch | link_import | live | canvas | feedback | draft_select | cleared | used | survey | other
    ts        TEXT NOT NULL,             -- ISO UTC
    prompt    TEXT NOT NULL,             -- user's words (title minus prefix)
    snapshot  TEXT NOT NULL              -- raw snapshot it was last seen in
);
CREATE INDEX IF NOT EXISTS turn_ts ON turn(ts);
CREATE INDEX IF NOT EXISTS turn_kind ON turn(kind);

CREATE TABLE IF NOT EXISTS conversation (
    id        TEXT PRIMARY KEY,          -- gemini.google.com/app/<id>
    source    TEXT NOT NULL,
    first_ts  TEXT, last_ts TEXT,
    n_turns   INTEGER,
    title     TEXT                       -- filled later by a model
);

-- Many-to-many: turns before a branch point belong to every branch.
CREATE TABLE IF NOT EXISTS turn_conv (
    turn_id   TEXT NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
    conv_id   TEXT NOT NULL,
    inferred  INTEGER NOT NULL DEFAULT 0, -- 1 = attached by time proximity, not by URL
    PRIMARY KEY (turn_id, conv_id)
);
CREATE INDEX IF NOT EXISTS turn_conv_conv ON turn_conv(conv_id);

CREATE TABLE IF NOT EXISTS answer (
    turn_id   TEXT NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
    draft_idx INTEGER NOT NULL,          -- >0 when the user regenerated
    html      TEXT NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (turn_id, draft_idx)
);

CREATE TABLE IF NOT EXISTS attachment (
    id          TEXT PRIMARY KEY,        -- turn_id:position
    turn_id     TEXT NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
    pos         INTEGER NOT NULL,
    orig_name   TEXT,                    -- name the user uploaded
    stored_name TEXT NOT NULL,           -- name in the Takeout
    path        TEXT,                    -- resolved file on disk, NULL if not found
    match       TEXT NOT NULL,           -- exact | no_ext | dup_suffix | missing
    mime        TEXT,
    caption     TEXT, ocr TEXT, transcript TEXT   -- filled by enrich/
);
CREATE INDEX IF NOT EXISTS attachment_turn ON attachment(turn_id);

CREATE TABLE IF NOT EXISTS canvas (
    turn_id   TEXT PRIMARY KEY REFERENCES turn(id) ON DELETE CASCADE,
    title     TEXT,
    content   TEXT
);

-- Explicit preference signals (thumbs, chosen draft), attached to the nearest preceding dialogue turn.
CREATE TABLE IF NOT EXISTS signal (
    turn_id    TEXT PRIMARY KEY REFERENCES turn(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    value      TEXT,
    target_id  TEXT                      -- dialogue turn it most likely refers to
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);

-- Trigram FTS works for Chinese substrings without a tokenizer.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_turn USING fts5(
    turn_id UNINDEXED, prompt, answer, extra, tokenize = 'trigram'
);
"""


def _backfill_frame_origins(con):
    key = "frame_archive_origins_v1"
    if con.execute("SELECT 1 FROM meta WHERE k=?",(key,)).fetchone():
        return
    archives = con.execute("""SELECT id,path,origin FROM attachment WHERE origin IN ('user','live_frame')
                              AND path IS NOT NULL AND lower(stored_name) LIKE '%.zip'""").fetchall()
    for aid,path,previous in archives:
        origin = "live_frame" if is_frame_archive(path, fallback=previous=="live_frame") else "user"
        if origin != previous:
            con.execute("UPDATE attachment SET origin=? WHERE id=?",(origin,aid))
    con.execute("INSERT INTO meta VALUES (?, '1')",(key,))


def _create_review_tables(con):
    statements = (
        """CREATE TABLE conv_summary (
            conv_id TEXT PRIMARY KEY REFERENCES conversation(id) ON DELETE CASCADE,
            summary TEXT NOT NULL,kinds TEXT NOT NULL,model TEXT NOT NULL,
            input_hash TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE digest (
            period TEXT PRIMARY KEY,level TEXT NOT NULL,body_md TEXT NOT NULL,model TEXT NOT NULL,
            input_hash TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE fact (
            id TEXT PRIMARY KEY,text TEXT NOT NULL,kind TEXT NOT NULL,date_from TEXT,date_to TEXT,
            confidence INTEGER NOT NULL CHECK(confidence BETWEEN 1 AND 3),status TEXT NOT NULL,
            superseded_by TEXT REFERENCES fact(id),verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE fact_evidence (
            fact_id TEXT NOT NULL REFERENCES fact(id) ON DELETE CASCADE,
            turn_id TEXT NOT NULL REFERENCES turn(id) ON DELETE CASCADE,quote TEXT NOT NULL,
            PRIMARY KEY(fact_id,turn_id))""",
        """CREATE TABLE fact_source (
            conv_id TEXT PRIMARY KEY REFERENCES conversation(id) ON DELETE CASCADE,
            input_hash TEXT NOT NULL,model TEXT NOT NULL,candidates TEXT NOT NULL,merged_hash TEXT)""",
    )
    for statement in statements:
        con.execute(statement)
    con.execute("PRAGMA user_version = 3")


def _migrate(con: sqlite3.Connection) -> None:
    with con:
        con.execute("BEGIN IMMEDIATE")
        version = con.execute("PRAGMA user_version").fetchone()[0]
        from .units import create_tables
        if version in (2,3,4):
            _backfill_frame_origins(con)
            if version==2:
                _create_review_tables(con)
            if version<4:
                create_tables(con)
            return
        if version != 0:
            raise ValueError(f"unsupported database version {version}")
        for column in ("origin", "enrich_model", "enrich_at", "enrich_error", "tags", "content_kind"):
            con.execute(f"ALTER TABLE attachment ADD COLUMN {column} TEXT")
        con.execute("""
            UPDATE attachment SET origin = CASE
                WHEN stored_name GLOB 'watermarked_img_*' THEN 'generated'
                WHEN lower(stored_name) LIKE '%.zip' AND EXISTS (
                    SELECT 1 FROM turn WHERE id=attachment.turn_id AND kind='live'
                ) THEN 'live_frame' ELSE 'user' END
        """)
        con.execute("""
            CREATE TABLE embedding (
                item_kind TEXT NOT NULL, item_id TEXT NOT NULL, model TEXT NOT NULL,
                dim INTEGER NOT NULL, input_hash TEXT NOT NULL, vec BLOB NOT NULL,
                PRIMARY KEY (item_kind, item_id, model)
            )
        """)
        con.execute("""
            CREATE TABLE search_cache (
                key TEXT PRIMARY KEY, kind TEXT NOT NULL, value TEXT NOT NULL, created_at TEXT NOT NULL
            )
        """)
        _backfill_frame_origins(con)
        _create_review_tables(con)
        create_tables(con)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        _migrate(con)
        con.execute("""CREATE TABLE IF NOT EXISTS fact_alias (
            id TEXT PRIMARY KEY,fact_id TEXT NOT NULL REFERENCES fact(id) ON DELETE CASCADE)""")
    except Exception:
        con.close()
        raise
    return con
