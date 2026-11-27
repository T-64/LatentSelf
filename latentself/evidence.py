"""Source text and citation checks shared by summaries, facts and writing."""
from dataclasses import dataclass
from hashlib import sha256
import json
import re

from .embed import DIALOGUE
from .sources import turn_order


@dataclass(frozen=True)
class SourceTurn:
    id: str
    ts: str
    prompt: str
    text: str


def source_turns(con,cid):
    rows = con.execute(f"""SELECT t.id,t.ts,t.prompt,a.caption FROM turn_conv tc JOIN turn t ON t.id=tc.turn_id
        LEFT JOIN attachment a ON a.turn_id=t.id AND a.origin IS NOT 'generated' AND a.caption IS NOT NULL
        WHERE tc.conv_id=? AND t.kind IN {DIALOGUE} ORDER BY {turn_order(con,cid)},a.pos""",(cid,))
    turns = {}
    for tid,ts,prompt,caption in rows:
        if tid not in turns:
            turns[tid] = (ts,prompt,[prompt])
        if caption:
            turns[tid][2].append("[附件] "+caption)
    return [SourceTurn(tid,ts,prompt,"\n".join(parts)) for tid,(ts,prompt,parts) in turns.items()]


def content_hash(*values):
    return sha256(json.dumps(values,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def original_quote(prompt,quote):
    """Return the exact original substring, allowing the model to change whitespace."""
    if not isinstance(quote,str) or not quote.strip() or len(quote)>120:
        return None
    if quote in prompt:
        return quote
    letters = [re.escape(ch) for ch in quote if not ch.isspace()]
    match = re.search(r"\s*".join(letters),prompt)
    if match and len(match[0])<=120:
        return match[0]
    return None


CITATION = re.compile(r"\[([ct]):([^\]\s]+)\]")


def references(markdown):
    return {f"{kind}:{key}" for kind,key in CITATION.findall(markdown)}


def cited_lines(markdown,allowed):
    """Retain headings and sourced prose. Count each invalid citation and unsourced line."""
    cleaned = []
    rejected = omitted = 0
    def legal(match):
        nonlocal rejected
        if match[0][1:-1] in allowed:
            return match[0]
        rejected += 1
        return ""
    for line in markdown.splitlines():
        line = CITATION.sub(legal,line).strip()
        if line and not line.startswith("#") and not references(line):
            omitted += 1
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip(),rejected,omitted
