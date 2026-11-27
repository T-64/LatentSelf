"""Personal analysis units and chronological sessions."""
from datetime import datetime,timedelta,timezone
import json
import random
import re

from .embed import DIALOGUE
from .exports import export_markdown


def create_tables(con):
    con.execute("ALTER TABLE turn ADD COLUMN is_paste INTEGER CHECK(is_paste IN (0,1))")
    for name,columns in {
        "episode":"id TEXT PRIMARY KEY,start_ts TEXT,end_ts TEXT,n_turns INTEGER",
        "episode_turn":"episode_id TEXT,turn_id TEXT,PRIMARY KEY(episode_id,turn_id)",
        "sae_run":"id TEXT PRIMARY KEY,reader TEXT,layer INTEGER,sae_repo TEXT,width INTEGER,topk INTEGER,precision TEXT,unit_kind TEXT,created_at TEXT,notes TEXT",
        "sae_act":"run_id TEXT,unit_id TEXT,feature INTEGER,act REAL,act_z REAL,argmax_tok INTEGER,PRIMARY KEY(run_id,unit_id,feature)",
        "sae_unit":"run_id TEXT,unit_id TEXT,n_tokens INTEGER,n_chunks INTEGER,fvu REAL,input_hash TEXT,PRIMARY KEY(run_id,unit_id)",
        "feature":"run_id TEXT,feature INTEGER,freq_self REAL,freq_base REAL,enrich_logodds REAL,enrich_z REAL,label TEXT,label_note TEXT,detect_score REAL,label_model TEXT,PRIMARY KEY(run_id,feature)",
        "baseline_unit":"id TEXT PRIMARY KEY,source TEXT,n_tokens INTEGER",
        "topic":"id TEXT PRIMARY KEY,resolution REAL,parent_id TEXT,name TEXT,size INTEGER,first_ts TEXT,last_ts TEXT,n_recur INTEGER,active_days INTEGER",
        "topic_member":"topic_id TEXT,conv_id TEXT,weight REAL,PRIMARY KEY(topic_id,conv_id)",
    }.items():
        con.execute(f"CREATE TABLE {name} ({columns})")
    con.execute("CREATE INDEX sae_act_feature ON sae_act(run_id,feature)")
    con.execute("PRAGMA user_version=4")


def is_paste(text):
    if re.search(r"(?:^|\n)\s*(?:作者|来源|原文链接|转自|转载自)\s*[:：]|\d+\s*人?赞同",text):
        return True
    if len(text)>1500 and (text.count("```")>=2 or len(re.findall(r"(?m)^\s*(?:#{1,6}\s|[-*>]\s|\d+[.)、]\s*)",text))>=2):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    return len(text)>800 and bool(letters) and sum(ch.isascii() for ch in letters)>0.8*len(letters)


def instant(ts):
    value = datetime.fromisoformat(ts if len(ts)>4 else ts+"-01-01")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def units(con,recompute=False):
    rows = sorted(con.execute(f"SELECT id,ts,prompt,is_paste FROM turn WHERE kind IN {DIALOGUE}"),
                  key=lambda r:(instant(r["ts"]),r["id"]))
    groups = []
    previous = None
    with con:
        con.execute("DELETE FROM episode_turn")
        con.execute("DELETE FROM episode")
        for row in rows:
            now = instant(row["ts"])
            if previous is None or now-previous>timedelta(minutes=30):
                groups.append([])
            groups[-1].append(row)
            previous = now
            if recompute or row["is_paste"] is None:
                con.execute("UPDATE turn SET is_paste=? WHERE id=?",(int(is_paste(row["prompt"])),row["id"]))
        for group in groups:
            first,last = group[0],group[-1]
            con.execute("INSERT INTO episode VALUES (?,?,?,?)",(first["id"],first["ts"],last["ts"],len(group)))
            con.executemany("INSERT INTO episode_turn VALUES (?,?)",[(first["id"],row["id"]) for row in group])
    return dict(turns=len(rows),episodes=len(groups))


def review_paste(con,out,count,seed=42,answer=None):
    if count<1:
        raise ValueError("sample size must be positive")
    rows = list(con.execute(f"SELECT id,prompt,is_paste FROM turn WHERE kind IN {DIALOGUE} AND is_paste IS NOT NULL ORDER BY id"))
    sample = random.Random(seed).sample(rows,min(count,len(rows)))
    checked = correct = 0
    lines = []
    for i,row in enumerate(sample):
        key = "paste_review:"+row["id"]+":"+str(row["is_paste"])
        saved = con.execute("SELECT v FROM meta WHERE k=?",(key,)).fetchone()
        verdict = json.loads(saved[0]) if saved else None
        if i<20 and answer and verdict is None:
            reply = answer(f'{row["prompt"]}\n判定粘贴={row["is_paste"]}，正确？[y/n/回车跳过] ').lower().strip()
            if reply in ("y","n"):
                verdict = reply=="y"
                with con:
                    con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(key,json.dumps(verdict)))
        if i<20 and verdict is not None:
            checked += 1
            correct += verdict
        lines.append(f'{i+1}. 判定 {row["is_paste"]} [t:{row["id"]}]\n\n{row["prompt"]}')
    path = export_markdown(con,out/"review"/"paste.md","\n\n".join(lines))
    return dict(path=str(path),sample=len(sample),checked=checked,correct=correct,passed=checked==20 and correct>=18)
