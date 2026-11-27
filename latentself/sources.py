"""Resolve citations and preserve a source's own message order."""
from urllib.parse import quote


def conversation_url(con, cid):
    if not cid:
        return None
    if cid.startswith("aistudio:"):
        row = con.execute("SELECT url FROM aistudio_source WHERE conv_id=?", (cid,)).fetchone()
        return row[0] if row else None
    return "https://gemini.google.com/app/" + quote(cid, safe="")


def turn_order(con, cid):
    if cid.startswith("aistudio:") and con.execute(
            "SELECT 1 FROM sqlite_master WHERE name='aistudio_chunk'").fetchone():
        return "(SELECT min(chunk_index) FROM aistudio_chunk WHERE turn_id=t.id)"
    return "t.ts,t.id"
