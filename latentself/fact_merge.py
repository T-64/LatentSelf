"""Resolve dated fact candidates against the calibrated dense representation."""
from itertools import batched
import json

import numpy as np

from .backends.embed import normalize
from .evidence import content_hash
from .facts import fact_id

RELATIONS = ("same","new_replaces_old","old_replaces_new","different")
SCHEMA = {"type":"object","additionalProperties":False,"required":["relation"],
          "properties":{"relation":{"enum":list(RELATIONS)}}}


def fact_vectors(con,backend,sources):
    texts = dict(con.execute("SELECT id,text FROM fact"))
    for _,_,candidates,_ in sources:
        for item in json.loads(candidates):
            texts.setdefault(fact_id(item["text"]),item["text"])
    saved = dict(con.execute("SELECT item_id,input_hash FROM embedding WHERE item_kind='fact' AND model=?",(backend.model,)))
    change = [(fid,text,content_hash(text)) for fid,text in texts.items() if saved.get(fid)!=content_hash(text)]
    for batch in batched(change,32):
        vecs = normalize(backend.embed_texts([text for _,text,_ in batch]))
        with con:
            con.executemany("""INSERT INTO embedding(item_kind,item_id,model,dim,input_hash,vec)
                VALUES ('fact',?,?,?,?,?) ON CONFLICT(item_kind,item_id,model) DO UPDATE SET
                dim=excluded.dim,input_hash=excluded.input_hash,vec=excluded.vec""",
                [(fid,backend.model,len(vec),digest,vec.astype("<f2").tobytes())
                 for (fid,_,digest),vec in zip(batch,vecs,strict=True)])
    return len(change)


def similarity_match(con,llm,item,fid,ids,scores):
    """Return a canonical ID and relation. Examine every plausible match in order."""
    active = {r["id"]:r for r in con.execute("SELECT id,text,kind,date_from FROM fact WHERE status='active'")}
    neighbours = [(float(scores[i]),ids[i]) for i in np.flatnonzero(scores>=0.80) if ids[i] in active]
    for score,old_id in sorted(neighbours,key=lambda p:(-p[0],p[1])):
        old = active[old_id]
        if score>=0.92 and old["kind"]==item["kind"]:
            return old_id,"same"
        fields = ("text","kind","date_from")
        material = {"old":{k:old[k] for k in fields},"new":{k:item.get(k) for k in fields}}
        vote = llm.chat([{"role":"system","content":"判断两个关于用户的事实。same 表示同一事实，"
            "new_replaces_old 或 old_replaces_new 表示同一属性的状态在时间上取代，different 表示独立。"
            "只根据内容和日期判断，同时成立的事实应独立。"},{"role":"user","content":json.dumps(material,ensure_ascii=False)}],
            json_schema=SCHEMA,max_tokens=60)["relation"]
        if vote not in RELATIONS:
            raise ValueError("unknown fact relation")
        if vote!="different":
            return old_id,vote
    return fid,"different"


def apply_candidate(con,llm,item,ids,pos,vectors):
    fid = fact_id(item["text"])
    found = con.execute("""SELECT id FROM fact WHERE id=? UNION ALL
                        SELECT fact_id FROM fact_alias WHERE id=? LIMIT 1""",(fid,fid)).fetchone()
    if found:
        match,relation = found[0],"same"
    else:
        match,relation = similarity_match(con,llm,item,fid,ids,vectors@vectors[pos[fid]])
    canonical = match if relation=="same" else fid
    if relation!="same":
        con.execute("""INSERT INTO fact(id,text,kind,date_from,date_to,confidence,status)
            VALUES (?,?,?,?,?,?,'active')""",(fid,item["text"],item["kind"],item["date_from"],item["date_to"],item["confidence"]))
    changes = con.total_changes
    for tid,quote in item["evidence"].items():
        con.execute("""INSERT INTO fact_evidence VALUES (?,?,?) ON CONFLICT(fact_id,turn_id) DO UPDATE
            SET quote=excluded.quote WHERE length(excluded.quote)>length(quote) OR NOT EXISTS (
                SELECT 1 FROM turn WHERE id=excluded.turn_id AND instr(prompt,fact_evidence.quote)>0)""",(canonical,tid,quote))
    if con.total_changes>changes:
        con.execute("""UPDATE fact SET verified=0,status=CASE WHEN status='rejected' THEN 'active' ELSE status END
                    WHERE id=?""",(canonical,))
    con.execute("UPDATE fact SET confidence=max(confidence,?) WHERE id=?",(item["confidence"],canonical))
    if relation=="same" and fid!=canonical:
        con.execute("INSERT OR IGNORE INTO fact_alias VALUES (?,?)",(fid,canonical))
    if relation in ("new_replaces_old","old_replaces_new"):
        older,newer = (match,fid) if relation=="new_replaces_old" else (fid,match)
        dates = dict(con.execute("SELECT id,date_from FROM fact WHERE id IN (?,?)",(older,newer)))
        if dates[older] and dates[newer] and dates[older]>dates[newer]:
            older,newer = newer,older
        con.execute("""UPDATE fact SET status='superseded',date_to=?,superseded_by=? WHERE id=?""",
                    (dates[newer],newer,older))
    return relation


def merge(con,llm,backend,vectors_only=False,emit=print):
    sources = con.execute("""SELECT s.conv_id,s.input_hash,s.candidates,s.merged_hash FROM fact_source s
        JOIN conversation c ON c.id=s.conv_id ORDER BY c.first_ts,s.conv_id""").fetchall()
    count = fact_vectors(con,backend,sources)
    stats = dict(success=0,skipped=0,vectors_written=count,same=0,different=0,new_replaces_old=0,old_replaces_new=0)
    if vectors_only:
        return stats
    rows = con.execute("SELECT item_id,vec FROM embedding WHERE item_kind='fact' AND model=? ORDER BY item_id",
                       (backend.model,)).fetchall()
    ids = [r[0] for r in rows]
    pos = {fid:i for i,fid in enumerate(ids)}
    vecs = normalize([np.frombuffer(r[1],dtype="<f2") for r in rows]) if rows else None
    for cid,source_hash,payload,completed in sources:
        key = content_hash(source_hash,backend.model,llm.model)
        if completed==key:
            stats["skipped"] += 1
            continue
        with con:
            con.execute("BEGIN IMMEDIATE")
            current = con.execute("SELECT input_hash,merged_hash FROM fact_source WHERE conv_id=?",(cid,)).fetchone()
            if not current or current[0]!=source_hash or current[1]==key:
                stats["skipped"] += 1
                continue
            for item in json.loads(payload):
                relation = apply_candidate(con,llm,item,ids,pos,vecs)
                stats[relation] += 1
            con.execute("UPDATE fact_source SET merged_hash=? WHERE conv_id=?",(key,cid))
        stats["success"] += 1
        emit(f"success {stats['success']}")
    return stats
