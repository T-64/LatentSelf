"""Ground film mentions and explicit preferences in source turns."""
from collections import deque
from datetime import datetime,timezone
import json
import random
import unicodedata
from uuid import uuid4

from .catalog import current_id,dumps,set_state
from .embed import DIALOGUE
from .evidence import content_hash
from .exports import export_markdown

SCHEMA = dict(type="object",additionalProperties=False,required=["status","sentiment"],properties={
    "status":dict(enum=["watched","wanted","mentioned_only"]),"sentiment":dict(enum=[-1,0,1])})


class AliasIndex:
    """Match every eligible alias in one pass over each prompt."""
    def __init__(self,pairs):
        self.edges = [{}]
        self.outputs = [[]]
        self.fail = [0]
        for iid,alias in pairs:
            word = unicodedata.normalize("NFKC",alias).casefold().strip()
            if len(word)<3:
                continue
            node = 0
            for ch in word:
                if ch not in self.edges[node]:
                    self.edges[node][ch] = len(self.edges)
                    self.edges.append({})
                    self.outputs.append([])
                    self.fail.append(0)
                node = self.edges[node][ch]
            self.outputs[node].append((iid,len(word)))
        queue = deque(self.edges[0].values())
        while queue:
            parent = queue.popleft()
            for ch,child in self.edges[parent].items():
                queue.append(child)
                fallback = self.fail[parent]
                while fallback and ch not in self.edges[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.edges[fallback].get(ch,0)
                self.outputs[child].extend(self.outputs[self.fail[child]])

    def match(self,prompt):
        text = unicodedata.normalize("NFKC",prompt).casefold()
        node = 0
        matches = {}
        for offset,ch in enumerate(text,1):
            while node and ch not in self.edges[node]:
                node = self.fail[node]
            node = self.edges[node].get(ch,0)
            for iid,length in self.outputs[node]:
                matches.setdefault(iid,text[max(0,offset-length-350):offset+350])
        return matches


def mention_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS item_mention(
        item_id TEXT,turn_id TEXT,status TEXT,sentiment INTEGER,input_hash TEXT,owner_correct INTEGER,
        PRIMARY KEY(item_id,turn_id))""")


def detect(con,llm,limit=None):
    mention_table(con)
    matcher = AliasIndex(con.execute("SELECT item_id,alias FROM item_alias"))
    titles = dict(con.execute("SELECT id,coalesce(title_zh,title) FROM item"))
    stats = dict(success=0,skipped=0,limited=0,failed=0,failures=[])
    for tid,text,ts in con.execute(f"SELECT id,prompt,ts FROM turn WHERE kind IN {DIALOGUE} ORDER BY ts,id"):
        for iid,excerpt in matcher.match(text).items():
            h = content_hash("film-mention-v1",llm.model,titles[iid],excerpt)
            if con.execute("SELECT 1 FROM item_mention WHERE item_id=? AND turn_id=? AND input_hash=?",(iid,tid,h)).fetchone():
                stats["skipped"] += 1
                continue
            if limit is not None and stats["success"]+stats["failed"]>=limit:
                stats["limited"] += 1
                continue
            try:
                verdict = llm.chat([dict(role="system",content="根据用户自己的原话判定是否已看或想看指定影片。讨论影片或他人行为归 mentioned_only。"),
                                    dict(role="user",content=dumps(dict(film=titles[iid],text=excerpt)))],json_schema=SCHEMA,max_tokens=100)
                status,sentiment = verdict["status"],verdict["sentiment"]
                if status not in ("watched","wanted","mentioned_only") or sentiment not in (-1,0,1):
                    raise ValueError("invalid mention")
                with con:
                    con.execute("INSERT OR REPLACE INTO item_mention VALUES (?,?,?,?,?,NULL)",(iid,tid,status,sentiment,h))
                    rows = con.execute("""SELECT m.turn_id,m.status,m.sentiment,t.ts FROM item_mention m JOIN turn t ON t.id=m.turn_id
                        WHERE m.item_id=? AND m.status!='mentioned_only' ORDER BY t.ts,m.turn_id""",(iid,)).fetchall()
                    con.execute("DELETE FROM item_state WHERE item_id=? AND source='chat_mention'",(iid,))
                    if rows:
                        latest = rows[-1]
                        set_state(con,iid,latest["status"],latest["sentiment"],"chat_mention",[r[0] for r in rows],latest["ts"])
                stats["success"] += 1
            except OSError as exc:
                stats["interrupted"] = type(exc).__name__
                return stats
            except (ValueError,KeyError) as exc:
                stats["failed"] += 1
                stats["failures"].append(dict(item_id=iid,turn_id=tid,error=str(exc)))
    return stats


def review(con,out,answer=None,seed=42):
    mention_table(con)
    rows = con.execute("""SELECT m.*,coalesce(i.title_zh,i.title) AS title,t.prompt FROM item_mention m
        JOIN item i ON i.id=m.item_id JOIN turn t ON t.id=m.turn_id WHERE m.status!='mentioned_only'
        ORDER BY m.item_id,m.turn_id""").fetchall()
    sample = random.Random(seed).sample(rows,min(30,len(rows)))
    checked = correct = 0
    lines = []
    for row in sample:
        text = f'{row["title"]} · {row["status"]} · {row["sentiment"]} [t:{row["turn_id"]}]\n\n{row["prompt"]}'
        lines.append(text)
        verdict = row["owner_correct"]
        if verdict is None and answer:
            reply = answer(text+"\n判定正确？[y/n/回车跳过] ").strip().lower()
            if reply in ("y","n"):
                verdict = reply=="y"
                with con:
                    con.execute("UPDATE item_mention SET owner_correct=? WHERE item_id=? AND turn_id=?",
                                (verdict,row["item_id"],row["turn_id"]))
        if verdict is not None:
            checked += 1
            correct += bool(verdict)
    path = export_markdown(con,out/"review"/"watched.md","# 观影识别核对\n\n"+"\n\n".join(lines))
    return dict(path=str(path),sample=len(sample),checked=checked,correct=correct,passed=checked==30 and correct>=27)


def feedback(con,identifier,action):
    identifier = current_id(con,identifier)
    row = con.execute("SELECT id FROM item WHERE id=?",(identifier,)).fetchone()
    if not row:
        matches = con.execute("""SELECT DISTINCT id FROM item LEFT JOIN item_alias a ON a.item_id=item.id
            WHERE a.alias=? OR title=? OR title_zh=? OR original_title=?""",(identifier,)*4).fetchall()
        if len(matches)!=1:
            raise ValueError("use an unambiguous film ID")
        row = matches[0]
    iid = row[0]
    prior = con.execute("SELECT rating FROM item_state WHERE item_id=?",(iid,)).fetchone()
    rating = dict(like=1,dislike=-1).get(action,prior[0] if prior else None)
    status = {"like":"watched","dislike":"watched","watched":"watched","wanted":"wanted","dismiss":"dismissed"}[action]
    stamp = datetime.now(timezone.utc).isoformat()
    sid = "rec-feedback-"+uuid4().hex
    with con:
        set_state(con,iid,status,rating,"feedback",when=stamp)
        con.execute("""INSERT INTO turn(id,source,kind,ts,prompt,snapshot) VALUES (?,'latentself','feedback',?,'','feedback')""",(sid,stamp))
        con.execute("INSERT INTO signal VALUES (?,'rec_feedback',?,NULL)",(sid,dumps(dict(item=iid,action=action))))
    return dict(item=iid,status=status,rating=rating)
