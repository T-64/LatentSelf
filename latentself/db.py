"""SQLite schema. One file holds the unified model, full-text index and later vectors/features."""
import sqlite3
from pathlib import Path

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


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con
