"""Frozen, source-stratified P1 review frames. All persistent rows are private."""
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import random
import sqlite3

import numpy as np

from .backends.embed import TEXT, VL, Reranker
from .backends.llm import LocalLLM
from .evaluation import evaluation_db, evaluation_vector
from .search import VectorIndex, _judge, _previews, expand, matching_turns, rank_conversations, read_policy
from .sources import conversation_url

QUOTAS = {"S1": 4, "S2": 6, "S3": 6}


def open_review(path, legacy=None):
    db = evaluation_db(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS frame(q,cid,source,tier,routes,excerpt,url,PRIMARY KEY(q,cid));
        CREATE TABLE IF NOT EXISTS sample(q,cid,position INTEGER,PRIMARY KEY(q,cid));
        CREATE TABLE IF NOT EXISTS decision(q,cid,answer CHECK(answer IN ('y','n','s')),PRIMARY KEY(q,cid));
        CREATE TABLE IF NOT EXISTS target(q PRIMARY KEY,path);
        CREATE TABLE IF NOT EXISTS prediction(q,config,rank INTEGER,cid,PRIMARY KEY(q,config,cid));
        CREATE TABLE IF NOT EXISTS strategy(config PRIMARY KEY,model,threshold REAL);
        CREATE TABLE IF NOT EXISTS frame_stats(q PRIMARY KEY,details);
        CREATE TABLE IF NOT EXISTS review_meta(k PRIMARY KEY,v);
    """)
    if legacy and Path(legacy).is_file() and not db.execute("SELECT 1 FROM review_meta WHERE k='legacy'").fetchone():
        with closing(sqlite3.connect(Path(legacy).absolute().as_uri()+"?mode=ro",uri=True)) as old, db:
            for q,cid,relevant,photo in old.execute("""SELECT l.* FROM label l LEFT JOIN consensus_label c USING(q,cid)
                    WHERE c.cid IS NULL"""):
                db.execute("INSERT OR IGNORE INTO decision VALUES(?,?,?)",(q,cid,"y" if relevant else "n"))
                if photo:
                    db.execute("INSERT OR IGNORE INTO target VALUES(?,?)",(q,photo))
            db.executemany("INSERT OR IGNORE INTO query_vector VALUES(?,?,?,?)",old.execute("SELECT * FROM query_vector"))
            db.execute("INSERT INTO review_meta VALUES('legacy',1)")
    return db


def carry_decisions(db,path):
    if db.execute("SELECT 1 FROM review_meta WHERE k='sealed'").fetchone():
        raise ValueError("use a fresh frame to carry decisions forward")
    with closing(sqlite3.connect(Path(path).absolute().as_uri()+"?mode=ro",uri=True)) as old, db:
        db.executemany("INSERT OR REPLACE INTO decision VALUES(?,?,?)",old.execute("SELECT q,cid,answer FROM decision"))
        db.executemany("INSERT OR REPLACE INTO target VALUES(?,?)",old.execute("SELECT q,path FROM target"))


def allocate(groups, existing, quota):
    """Constrained largest remainder, including existing owner labels and the AI Studio minimum."""
    total = sum(map(len,groups.values()))
    slots = min(total,quota)
    alloc = {s:len(set(ids) & existing) for s,ids in groups.items()}
    if groups.get("aistudio"):
        alloc["aistudio"] = max(1,alloc["aistudio"])
    while sum(alloc.values()) < slots:
        eligible = [s for s in groups if alloc[s]<len(groups[s])]
        winner = max(eligible,key=lambda s:(slots*len(groups[s])/total-alloc[s],s))
        alloc[winner] += 1
    return alloc


def draw(db, queries, seed=42):
    selected = []
    for row in queries:
        q = row["q"]
        known = {r[0] for r in db.execute("SELECT cid FROM decision WHERE q=?",(q,))}
        for tier,quota in QUOTAS.items():
            groups = defaultdict(list)
            for cid,source in db.execute("SELECT cid,source FROM frame WHERE q=? AND tier=? ORDER BY cid",(q,tier)):
                groups[source].append(cid)
            for source,n in allocate(groups,known,quota).items():
                members = groups[source]
                prior = [cid for cid in members if cid in known]
                rng = random.Random(json.dumps([seed,q,tier,source],ensure_ascii=False))
                picks = prior + rng.sample([cid for cid in members if cid not in known],n-len(prior))
                selected.extend((q,cid) for cid in picks)
    if len(selected)>180:
        raise ValueError("retained owner decisions would exceed the 180-item limit")
    random.Random(seed).shuffle(selected)
    db.executemany("INSERT INTO sample VALUES(?,?,?)",[(q,cid,i) for i,(q,cid) in enumerate(selected,1)])


def candidate_frame(con, db, rows, backends, *, llm=None, seed=42, progress=lambda s:None):
    if db.execute("SELECT 1 FROM review_meta WHERE k='sealed'").fetchone():
        raise ValueError("this review is frozen; use another --review-db for a new frame")
    if len({r["q"] for r in rows})!=len(rows):
        raise ValueError("duplicate evaluation query")
    llm = llm or LocalLLM()
    by_model = {b.model:b for b in backends}
    indices = {m:VectorIndex(con,by_model[m],("turn",)) for m in (VL,TEXT)}
    images = VectorIndex(con,by_model[VL],("attachment",))
    multimodal = VectorIndex(con,by_model[VL],("turn","attachment"))
    if not indices[VL].ids or set(indices[VL].ids)!=set(indices[TEXT].ids):
        raise ValueError("paired text candidates must cover the same turns")
    if con.execute("""SELECT 1 FROM embedding a JOIN embedding b USING(item_kind,item_id)
            WHERE a.item_kind='turn' AND a.model=? AND b.model=? AND a.input_hash<>b.input_hash LIMIT 1""",(VL,TEXT)).fetchone():
        raise ValueError("paired inputs differ; refresh the candidate index")
    all_ids = {r[0] for r in con.execute("SELECT id FROM turn")}
    photos = {r[0] for r in con.execute("""SELECT turn_id FROM attachment WHERE path IS NOT NULL
                        AND (mime LIKE 'image/%' OR origin='live_frame')""")}
    membership = [(r[0],r[1]) for r in con.execute("SELECT conv_id,turn_id FROM turn_conv")]
    convs = dict(con.execute("SELECT id,source FROM conversation"))
    policy = read_policy(con)
    data = []
    for row in rows:
        q = row["q"]
        eligible = photos if row.get("photo") else all_ids
        terms = list(dict.fromkeys([q,*expand(con,llm,q)]))
        loose = {t for term in terms for t in matching_turns(con,term,True) if t in eligible}
        keywords = Counter(t for term in terms for t in matching_turns(con,term) if t in eligible)
        vectors = {m:evaluation_vector(db,b,q) for m,b in by_model.items()}
        scores = {m:idx.scores(q,eligible,vectors[m]) for m,idx in indices.items()}
        visual = images.scores(q,eligible,vectors[VL])
        combined = multimodal.scores(q,eligible,vectors[VL])
        data.append((q,terms,loose,keywords,scores,visual,combined))
    # Freeze outputs for every threshold considered during model selection, before drawing.
    from itertools import chain
    thresholds = {}
    for m in (VL,TEXT):
        values = np.fromiter(chain.from_iterable(d[4][m].values() for d in data),dtype=float)
        grid = set(map(float,np.quantile(values,np.linspace(0,1,11)))) if values.size else set()
        if "threshold" in policy:
            grid.add(float(policy["threshold"]))
        thresholds[m] = [None,*sorted(grid)]
    for number,(q,terms,loose,keywords,scores,visual,combined) in enumerate(data,1):
        routes, hits = defaultdict(set),defaultdict(dict)
        def record(tids,route):
            for cid,tid in membership:
                if tid in tids:
                    routes[cid].add(route)
                    hits[cid][tid] = (1,{"keyword"} if route=="keyword" else {"semantic"})
        record(loose,"keyword")
        def ordered(values):
            return sorted(values,key=lambda tid:(-values[tid],tid))
        visual_top = set(ordered(visual)[:300])
        record(visual_top,VL)
        predictions = []
        configs = []
        discarded = {}
        for model,values in scores.items():
            top = set(ordered(values)[:300])
            record(top,model)
            combined_top = set(ordered(combined)[:300]) if model==VL else set()
            if model==VL:
                record(combined_top,VL)
            for n,tau in enumerate(thresholds[model]):
                key = ("vl" if model==VL else "dual")+f"-{n}"
                if model==VL:
                    keep = {t:v for t,v in combined.items() if t in combined_top or (tau is not None and v>=tau)}
                else:
                    keep = {t:v for t,v in values.items() if t in top or (tau is not None and v>=tau)}
                    pure_ranked,pure_groups = rank_conversations(keywords,keep,membership)
                    pure_key = f"text-{n}"
                    configs.append((pure_key,model,tau))
                    discarded[pure_key] = max(0,len(pure_ranked)-200)
                    for rank,(_,cid) in enumerate(pure_ranked[:200],1):
                        predictions.append((q,pure_key,rank,cid))
                        if cid not in hits:
                            hits[cid].update(pure_groups[cid])
                        if any(t in keep for t in pure_groups[cid]):
                            routes[cid].add(model)
                    for t,v in visual.items():
                        if t in visual_top or (tau is not None and v>=tau):
                            keep[t] = max(v,keep.get(t,-1))
                ranked,groups = rank_conversations(keywords,keep,membership)
                configs.append((key,model,tau))
                discarded[key] = max(0,len(ranked)-200)
                for rank,(_,cid) in enumerate(ranked[:200],1):
                    predictions.append((q,key,rank,cid))
                    # Alternate thresholds of one model are not independent retrieval routes.
                    if cid not in hits:
                        hits[cid].update(groups[cid])
                    for tid in groups[cid]:
                        if tid in keep:
                            if model==VL or (tid in values and (tid in top or
                                                               tau is not None and values[tid]>=tau)):
                                routes[cid].add(model)
                            if model==TEXT and tid in visual and (tid in visual_top or
                                                                 tau is not None and visual[tid]>=tau):
                                routes[cid].add(VL)
                    if not routes[cid]:
                        raise ValueError("configuration output has no retrieval provenance")
        for cid, in db.execute("SELECT cid FROM decision WHERE q=?",(q,)):
            if cid not in convs:
                raise ValueError("previous owner label refers to unavailable conversation")
            if cid not in routes:
                routes[cid].add("prior-owner")
                hits[cid].update({t:(1,{"semantic"}) for c,t in membership if c==cid})
        review_stats = {"judge_prompt_chars_omitted":0}
        excerpts = {cid:_previews(con,cid,group,terms,True,review_stats)[1]
                    for cid,group in hits.items()}
        singles = {cid:excerpt for cid,excerpt in excerpts.items() if len(routes[cid])==1}
        from .search import _key,cached
        cold = sum(cached(con,_key("judge-v1",llm.model,{},q,cid,v)) is None for cid,v in singles.items())
        progress(dict(query=number,pool=len(routes),singleton=len(singles),new_judgments=cold))
        verdicts = _judge(con,llm,q,singles,{},Reranker())
        with db:
            review_stats.update(config_candidates_omitted=discarded,keyword_turns=len(loose),
                                vector_top300_omitted={m:max(0,len(v)-300) for m,v in scores.items()},
                                visual_top300_omitted=max(0,len(visual)-300))
            db.execute("INSERT OR REPLACE INTO frame_stats VALUES(?,?)",(q,json.dumps(review_stats)))
            db.execute("DELETE FROM frame WHERE q=?",(q,))
            db.execute("DELETE FROM prediction WHERE q=?",(q,))
            db.executemany("INSERT OR REPLACE INTO strategy VALUES(?,?,?)",configs)
            db.executemany("INSERT INTO prediction VALUES(?,?,?,?)",predictions)
            for cid,rr in routes.items():
                tier = "S1" if len(rr)>=2 else ("S2" if verdicts[cid]["relevant"] else "S3")
                db.execute("INSERT INTO frame VALUES(?,?,?,?,?,?,?)",(q,cid,convs[cid],tier,json.dumps(sorted(rr)),
                           excerpts[cid],conversation_url(con,cid)))
        progress(dict(query=number,limits=review_stats))
    with db:
        draw(db,rows,seed)
        db.execute("INSERT INTO review_meta VALUES('sealed',?)",(json.dumps(rows,sort_keys=True),))
        db.execute("INSERT INTO review_meta VALUES('seed',?)",(seed,))


def check_frame(db,rows):
    row = db.execute("SELECT v FROM review_meta WHERE k='sealed'").fetchone()
    if not row or json.loads(row[0])!=rows:
        raise ValueError("generate the stratified review for the current query file first")


def status(db,rows):
    check_frame(db,rows)
    result = []
    for n,row in enumerate(rows,1):
        q = row["q"]
        counts = [dict(source=s,tier=t,pool=p,sample=k,labeled=y,skipped=skip,pending=k-y-skip)
            for s,t,p,k,y,skip in db.execute("""SELECT f.source,f.tier,count(*),count(a.cid),
             sum(CASE WHEN a.cid IS NOT NULL AND d.answer IN ('y','n') THEN 1 ELSE 0 END),
             sum(CASE WHEN a.cid IS NOT NULL AND d.answer='s' THEN 1 ELSE 0 END)
             FROM frame f LEFT JOIN sample a USING(q,cid) LEFT JOIN decision d USING(q,cid)
             WHERE f.q=? GROUP BY f.source,f.tier ORDER BY f.tier,f.source""",(q,))]
        present = {(r["source"],r["tier"]) for r in counts}
        counts.extend(dict(source=s,tier=t,pool=0,sample=0,labeled=0,skipped=0,pending=0)
                      for s in ("gemini","aistudio") for t in QUOTAS if (s,t) not in present)
        counts.sort(key=lambda r:(r["tier"],r["source"]))
        result.append(dict(number=n,strata=counts,not_sampled=sum(x["pool"]-x["sample"] for x in counts),photo_target=(bool(db.execute("SELECT 1 FROM target WHERE q=?",(q,)).fetchone())
                                                               if row.get("photo") else None)))
    return dict(queries=result,totals={k:sum(s[k] for r in result for s in r["strata"])
                for k in ("pool","sample","labeled","skipped","pending")})


def label_batch(db,rows,limit=30,ask=input,emit=print):
    check_frame(db,rows)
    if limit < 1:
        raise ValueError("batch must be positive")
    total = db.execute("SELECT count(*) FROM sample").fetchone()[0]
    todo = db.execute("""SELECT s.position,s.q,s.cid,f.source,f.excerpt,f.url FROM sample s JOIN frame f USING(q,cid)
                         LEFT JOIN decision d USING(q,cid) WHERE d.answer IS NULL ORDER BY s.position LIMIT ?""",(limit,)).fetchall()
    done = 0
    for position,q,cid,source,excerpt,url in todo:
        emit(f"\n清单 {position}/{total} · 查询：{q}\n来源：{source}\n{excerpt}\n{url}")
        while (choice:=ask("相关？ y 是 / n 否 / s 跳过 / q 暂停：").strip().lower()) not in ("y","n","s","q"):
            emit("请输入 y、n、s 或 q。")
        if choice=="q":
            break
        with db:
            db.execute("INSERT OR IGNORE INTO decision VALUES(?,?,?)",(q,cid,choice))
        done += 1
    return done


def set_photo(con,db,rows,number,path):
    check_frame(db,rows)
    if not 1<=number<=len(rows) or not rows[number-1].get("photo"):
        raise ValueError("choose a photo query number")
    path = Path(path).expanduser().absolute()
    indexed = con.execute("""SELECT path FROM attachment WHERE path IN (?,?)
                AND (mime LIKE 'image/%' OR origin='live_frame')
                ORDER BY (path=?) DESC LIMIT 1""",(str(path),str(path.resolve()),str(path))).fetchone()
    if not path.is_file() or not indexed:
        raise ValueError("target must be an available indexed photo")
    with db:
        db.execute("INSERT INTO target VALUES(?,?) ON CONFLICT(q) DO UPDATE SET path=excluded.path",(rows[number-1]["q"],indexed[0]))
