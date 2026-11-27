"""Private query pools, resumable owner labels and per-query acceptance metrics."""
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import numpy as np

from .backends.embed import Embedder, TEXT, VL
from .backends.llm import LocalLLM
from .search import VectorIndex, _judge, _previews, expand, matching_turns, search
from .backends.embed import Reranker


def read_queries(path):
    # Psych is in the macOS system Ruby. It handles YAML quoting, inline
    # mappings and comments without adding a Python dependency.
    data = subprocess.run(["/usr/bin/ruby","-ryaml","-rjson","-e",
                           "puts JSON.generate(YAML.safe_load(STDIN.read))"],
                          input=Path(path).read_text(),text=True,capture_output=True,check=True)
    rows = json.loads(data.stdout)["queries"]
    if not rows or any(not isinstance(r.get("q"),str) or not r["q"].strip() for r in rows):
        raise ValueError("every evaluation query requires a nonempty q")
    return rows


def evaluation_db(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS pool (
            q TEXT, cid TEXT, routes TEXT NOT NULL, excerpt TEXT NOT NULL, PRIMARY KEY(q,cid));
        CREATE TABLE IF NOT EXISTS photo_review (
            q TEXT, cid TEXT, excerpt TEXT NOT NULL, PRIMARY KEY(q,cid));
        CREATE TABLE IF NOT EXISTS label (
            q TEXT, cid TEXT, relevant INTEGER NOT NULL, photo TEXT, PRIMARY KEY(q,cid));
        CREATE TABLE IF NOT EXISTS consensus_label (
            q TEXT, cid TEXT, PRIMARY KEY(q,cid));
        CREATE TABLE IF NOT EXISTS query_vector (
            model TEXT, q TEXT, instruction TEXT, vec BLOB NOT NULL, PRIMARY KEY(model,q,instruction));
        CREATE TABLE IF NOT EXISTS rerank_batch (
            model TEXT, digest TEXT, scores TEXT NOT NULL, seconds REAL NOT NULL,
            PRIMARY KEY(model,digest));
    """)
    return con


def evaluation_vector(ev,backend,query):
    from .backends.embed import INSTRUCTION
    key = (backend.model,query,INSTRUCTION)
    row = ev.execute("SELECT vec FROM query_vector WHERE model=? AND q=? AND instruction=?",key).fetchone()
    if row:
        return np.frombuffer(row[0],dtype="<f4")
    vector = np.asarray(backend.embed_texts([query],kind="query")[0],dtype="<f4")
    with ev:
        ev.execute("INSERT OR IGNORE INTO query_vector VALUES (?,?,?,?)",(*key,vector.tobytes()))
    return vector


def prepare(con, ev, rows, backends, llm=None):
    llm = llm or LocalLLM()
    if {b.model for b in backends} != {VL,TEXT}:
        raise ValueError("evaluation requires both candidate models")
    indices = [VectorIndex(con,b,("turn",)) for b in backends]
    if any(not idx.ids for idx in indices) or set(indices[0].ids)!=set(indices[1].ids):
        raise ValueError("index the same evaluation turns with both candidate models first")
    mismatch = con.execute("""SELECT 1 FROM embedding a JOIN embedding b USING (item_kind,item_id)
        WHERE a.item_kind='turn' AND a.model=? AND b.model=? AND a.input_hash<>b.input_hash
        LIMIT 1""",(VL,TEXT)).fetchone()
    if mismatch:
        raise ValueError("evaluation text changed between model runs; rerun both comparison indexes")
    if any(r.get("photo") for r in rows):
        missing = con.execute("""SELECT count(*) FROM attachment a
            WHERE a.path IS NOT NULL AND (a.mime LIKE 'image/%' OR a.origin='live_frame')
            AND NOT EXISTS (SELECT 1 FROM embedding e WHERE e.item_kind='attachment'
                AND e.item_id=a.id AND e.model=?)""",(VL,)).fetchone()[0]
        if missing:
            raise ValueError(f"photo evaluation requires {missing} more visual attachment embeddings")
    coverage = set(indices[0].ids)
    from .embed import image_rows
    photo_turns = {r["turn_id"] for r in image_rows(con)}
    photo_indices = [VectorIndex(con,b,("turn","attachment")) if b.model == VL else idx
                     for b,idx in zip(backends,indices,strict=True)] if any(r.get("photo") for r in rows) else []
    membership = defaultdict(set)
    for tid,cid in con.execute("SELECT turn_id,conv_id FROM turn_conv"):
        membership[tid].add(cid)
    for row in rows:
        query = row["q"]
        terms = list(dict.fromkeys([query,*expand(con,llm,query)]))
        sources = {"keyword":set().union(*(set(matching_turns(con,t,True)) for t in terms))}
        eligible = coverage & photo_turns if row.get("photo") else coverage
        if row.get("photo"):
            sources["keyword"] &= photo_turns
        counts = {"judge_prompt_chars_omitted":0, "vector_turns_omitted":{}}
        for idx in (photo_indices if row.get("photo") else indices):
            scores,omitted = idx.recall(query,eligible,None,300,evaluation_vector(ev,idx.backend,query))
            counts["vector_turns_omitted"][idx.backend.model] = omitted
            sources[idx.backend.model] = set(scores)
        hits,routes = defaultdict(dict),defaultdict(set)
        for source,tids in sources.items():
            for tid in tids:
                for cid in membership[tid]:
                    hits[cid][tid] = (1,{"keyword"} if source=="keyword" else {"semantic"})
                    routes[cid].add(source)
        excerpts = {cid:_previews(con,cid,turns,terms,True,counts)[1]
                    for cid,turns in hits.items()}
        judgments = _judge(con,llm,query,excerpts,{},Reranker())
        with ev:
            ev.execute("""DELETE FROM label WHERE q=? AND cid IN
                          (SELECT cid FROM consensus_label WHERE q=?)""",(query,query))
            ev.execute("DELETE FROM consensus_label WHERE q=?",(query,))
            ev.execute("DELETE FROM pool WHERE q=?",(query,))
            for cid,excerpt in excerpts.items():
                if judgments[cid]["relevant"]:
                    routes[cid].add("strict")
                ev.execute("INSERT INTO pool VALUES (?,?,?,?)",(query,cid,json.dumps(sorted(routes[cid])),excerpt))
                if routes[cid] == {"keyword",TEXT,VL,"strict"} and not row.get("photo"):
                    inserted = ev.execute("INSERT OR IGNORE INTO label VALUES (?,?,1,NULL)",(query,cid)).rowcount
                    if inserted:
                        ev.execute("INSERT INTO consensus_label VALUES (?,?)",(query,cid))
        counts["consensus_labels"] = ev.execute("SELECT count(*) FROM consensus_label WHERE q=?",(query,)).fetchone()[0]
        counts["pending_owner_labels"] = ev.execute("""SELECT count(*) FROM pool p LEFT JOIN label l USING(q,cid)
            WHERE p.q=? AND l.cid IS NULL""",(query,)).fetchone()[0]
        yield {"candidates":len(excerpts),"covered_turns":len(coverage),**counts}


def prepare_photo_review(con, ev, rows, llm=None):
    """Queue photo keyword hits that will belong to the final union regardless of model rank."""
    from .embed import image_rows
    llm = llm or LocalLLM()
    photo_turns = {r["turn_id"] for r in image_rows(con)}
    for number,row in enumerate(rows,1):
        if not row.get("photo"):
            continue
        query = row["q"]
        terms = list(dict.fromkeys([query, *expand(con,llm,query)]))
        matches = {tid for term in terms for tid in matching_turns(con,term,True)} & photo_turns
        counts = {"judge_prompt_chars_omitted":0}
        hits = defaultdict(dict)
        for tid,cid in con.execute("SELECT turn_id,conv_id FROM turn_conv"):
            if tid in matches:
                hits[cid][tid] = (1,{"keyword"})
        with ev:
            for cid,turns in hits.items():
                _,excerpt = _previews(con,cid,turns,terms,True,counts)
                ev.execute("INSERT OR IGNORE INTO photo_review VALUES (?,?,?)",(query,cid,excerpt))
        yield {"query_number":number, "candidates":len(hits), **counts}


def label_batch(con, ev, rows, limit=30, ask=input, emit=print):
    done = 0
    for row in rows:
        query = row["q"]
        pending = ev.execute("""WITH candidates AS (
            SELECT * FROM pool WHERE q=:q
            UNION ALL
            SELECT e.q,e.cid,'["keyword"]',e.excerpt FROM photo_review e WHERE e.q=:q
              AND NOT EXISTS (SELECT 1 FROM pool p WHERE p.q=e.q AND p.cid=e.cid)
        ) SELECT p.* FROM candidates p LEFT JOIN label l USING(q,cid)
            WHERE l.cid IS NULL ORDER BY json_array_length(p.routes),p.cid""",{"q":query}).fetchall()
        for p in pending:
            if done >= limit:
                break
            # All disagreements are reviewed, including reranker/LLM negatives.
            emit(f"\n{query}\n{p['cid']}\n{p['excerpt']}")
            paths = []
            if row.get("photo"):
                paths = [r[0] for r in con.execute("""SELECT DISTINCT a.path FROM attachment a JOIN turn_conv tc
                    ON tc.turn_id=a.turn_id WHERE tc.conv_id=? AND a.path IS NOT NULL
                    AND (a.mime LIKE 'image/%' OR a.origin='live_frame') ORDER BY a.path""",(p["cid"],))]
                for i,path in enumerate(paths,1):
                    emit(f"{i}  {path}")
            prompt = "相关？ [y/n/q，输入 v 编号可在本机预览] " if paths else "相关？ [y/n/q] "
            while (choice:=ask(prompt).strip().lower()) not in ("y","n","q"):
                command = choice.split()
                if len(command)==2 and command[0]=="v" and command[1].isdecimal() and 1<=int(command[1])<=len(paths):
                    path = Path(paths[int(command[1])-1]).absolute()
                    result = subprocess.run(["/usr/bin/qlmanage","-p",str(path)],capture_output=True)
                    if result.returncode:
                        emit("预览未打开，可用上方路径在 Finder 中查看。")
                else:
                    emit("请按提示输入。")
            if choice=="q":
                return done
            photo = None
            if choice=="y" and paths:
                while True:
                    choice_id = ask("目标照片编号（无目标照片请回车）：").strip()
                    if not choice_id:
                        break
                    if choice_id.isdecimal() and 1<=int(choice_id)<=len(paths):
                        photo = paths[int(choice_id)-1]
                        break
            with ev:
                ev.execute("INSERT INTO label VALUES (?,?,?,?)",(query,p["cid"],choice=="y",photo))
            done += 1
    return done


def metrics(results, gold, photo=False):
    retrieved = {r["conv_id"] for r in results}
    relevant = {cid for cid,(positive,_) in gold.items() if positive}
    tp = len(retrieved & relevant)
    targets = {path for positive,path in gold.values() if positive and path}
    return {"recall":tp/len(relevant) if relevant else None,
            "precision":tp/len(retrieved) if retrieved else 0.0,
            "missed":sorted(relevant-retrieved),"false_positive":sorted((retrieved & gold.keys())-relevant),
            "unjudged":sorted(retrieved-gold.keys()),
            "photo_top5":(bool(targets & {p for r in results[:5] for p in r["images"]}) if targets else None)
                        if photo else None}


def evaluate(con,ev,rows,*,indexes=None,policy=None,llm=None):
    from .search import configured_indexes, read_policy
    policy = read_policy(con) if policy is None else policy
    indexes = configured_indexes(con,policy) if indexes is None else indexes
    for number,row in enumerate(rows,1):
        query = row["q"]
        pool = {r[0] for r in ev.execute("SELECT cid FROM pool WHERE q=?",(query,))}
        gold = {r[0]:(r[1],r[2]) for r in ev.execute("SELECT cid,relevant,photo FROM label WHERE q=?",(query,))}
        if not pool or pool-gold.keys():
            raise ValueError(f"query {number}: owner labels required ({len(pool-gold.keys())} pending)")
        for mode in ("fast","strict"):
            start = time.monotonic()
            out = search(con,query,mode=mode,images_only=bool(row.get("photo")),sort="score",
                         indexes=indexes,policy=policy,llm=llm)
            score = metrics(out["results"],gold,row.get("photo",False))
            valid = out["mode"]==mode and not out["notices"] and not score["unjudged"]
            yield {"query_number":number,"mode":mode,**score,"seconds":time.monotonic()-start,
                   "notices":out["notices"],"accepted":bool(valid and (score["recall"] or 0)>=0.95 and
                   (mode=="fast" or score["precision"]>=0.90) and (not row.get("photo") or score["photo_top5"]))}


def extend_pool(con,ev,rows):
    """Queue newly retrieved conversations after building the production index."""
    from .search import configured_indexes,read_policy
    policy = read_policy(con)
    indices = configured_indexes(con,policy)
    if not indices or any(not i.ids for i in indices):
        raise ValueError("full evaluation requires populated indexes")
    for number,row in enumerate(rows,1):
        out = search(con,row["q"],images_only=bool(row.get("photo")),sort="score",indexes=indices,policy=policy)
        if out["notices"]:
            raise ValueError(f"query {number}: incomplete retrieval: {out['notices']}")
        count = 0
        with ev:
            for r in out["results"]:
                material = "\n".join([r["first_prompt"],*(s["text"] for s in r["snippets"])])
                count += ev.execute("INSERT OR IGNORE INTO pool VALUES (?,?,?,?)",
                                    (row["q"],r["conv_id"],'["full-index"]',material)).rowcount
        yield number,count


def default_backends():
    import os
    return [Embedder(VL),Embedder(TEXT,os.environ.get("LATENTSELF_TEXT_EMBED_URL","http://127.0.0.1:8003/v1"))]


def index_pool(con,ev,rows,backend,llm):
    from .embed import index, image_rows
    seed = set()
    for row in rows:
        evaluation_vector(ev,backend,row["q"])
        for term in [row["q"], *expand(con,llm,row["q"])]:
            seed.update(matching_turns(con,term,True))
        if row.get("photo"):
            seed.update(r["turn_id"] for r in image_rows(con))
    if not seed:
        raise ValueError("empty comparison corpus")
    return index(con,backend,turn_ids=seed,include_images=any(r.get("photo") for r in rows),kinds=("turn",))
