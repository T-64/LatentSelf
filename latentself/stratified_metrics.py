"""Design-weighted P1 metrics. Each query, tier and source is an independent sampling cell."""
from collections import defaultdict
import json
import numpy as np

from .calibration import choose_model, cascade_bounds
from .backends.embed import TEXT, VL, Reranker
from .backends.llm import LocalLLM
from .search import _judge, _previews, configured_indexes, expand, matching_turns, rank_conversations
from .evaluation import evaluation_vector
from .stratified import check_frame, status


def cells(db,rows):
    check_frame(db,rows)
    if status(db,rows)["totals"]["pending"]:
        raise ValueError("owner review is incomplete")
    data = {}
    for row in rows:
        q = row["q"]
        for tier,source,size in db.execute("SELECT tier,source,count(*) FROM frame WHERE q=? GROUP BY tier,source",(q,)):
            labels = [(cid,answer=="y") for cid,answer in db.execute("""SELECT f.cid,d.answer FROM frame f
                JOIN sample s USING(q,cid) JOIN decision d USING(q,cid)
                WHERE f.q=? AND f.tier=? AND f.source=? AND d.answer IN ('y','n') ORDER BY f.cid""",(q,tier,source))]
            data[q,tier,source] = (size,labels)
    return data


def ratios(totals):
    tp,positive,retrieved = totals
    return dict(recall=float(tp/positive) if positive else None,
                precision=float(tp/retrieved) if retrieved else None)


def estimate(data,predicted,*,bootstrap=1000,seed=42):
    """Estimate ratios of HT totals. A skipped item is not a negative label."""
    rng = np.random.default_rng(seed)
    total = np.zeros(3)
    boot = np.zeros((bootstrap,3))
    empty = 0
    missed,fp = [],[]
    for (q,tier,source),(size,labels) in sorted(data.items()):
        if not labels:
            empty += 1
            continue
        rows = []
        for cid,positive in labels:
            hit = cid in predicted[q]
            rows.append((positive and hit,positive,hit))
            if positive and not hit:
                missed.append((q,cid))
            if hit and not positive:
                fp.append((q,cid))
        mat = np.array(rows,dtype=float)
        total += size*mat.mean(axis=0)
        if bootstrap:
            if size==len(labels):
                boot += size*mat.mean(axis=0)
            else:
                ix = rng.integers(len(labels),size=(bootstrap,len(labels)))
                boot += size*mat[ix].mean(axis=1)
    point = ratios(total) if data and not empty else dict(recall=None,precision=None)
    interval = {}
    for i,name in enumerate(("recall","precision"),1):
        valid = boot[:,i]>0
        interval[name] = (np.quantile(boot[valid,0]/boot[valid,i],[.025,.975]).tolist()
                          if point[name] is not None and valid.any() else None)
    return dict(**point,ci95=interval,unrepresented_cells=empty,missed=missed,false_positive=fp)


def frame_guard(db,q,ids):
    known = {r[0] for r in db.execute("SELECT cid FROM frame WHERE q=?",(q,))}
    outside = set(ids)-known
    if outside:
        raise ValueError(f"{len(outside)} outputs outside the frozen frame; repool before scoring")


def predictions(db,rows,key):
    return {r["q"]:[x[0] for x in db.execute("SELECT cid FROM prediction WHERE q=? AND config=? ORDER BY rank",
                                              (r["q"],key))] for r in rows}


def breakdown(data,rows,predicted,mode):
    for r in rows:
        if r["q"] not in predicted:
            raise ValueError("missing query output")
    text = {r["q"] for r in rows if not r.get("photo")}
    pooled = estimate({k:v for k,v in data.items() if k[0] in text},predicted)
    gates = dict(recall=.95) if mode=="fast" else dict(recall=.95,precision=.90)
    pooled["meets_text_gate"] = all(pooled[k] is not None and pooled[k]>=v for k,v in gates.items())
    pooled["lower_bound_below_gate"] = [k for k,v in gates.items() if pooled["ci95"][k] and pooled["ci95"][k][0]<v]
    return dict(text=pooled,by_query=[dict(number=i,**estimate({k:v for k,v in data.items() if k[0]==r["q"]},predicted))
                                      for i,r in enumerate(rows,1)],
                by_source={s:estimate({k:v for k,v in data.items() if k[0] in text and k[2]==s},predicted)
                           for s in sorted({k[2] for k in data})})


def select_config(db,data,rows):
    text = {r["q"] for r in rows if not r.get("photo")}
    data = {k:v for k,v in data.items() if k[0] in text}
    if not data:
        raise ValueError("text query labels are required")
    best = {}
    for key,model,tau in db.execute("SELECT config,model,threshold FROM strategy ORDER BY config"):
        family = key.split("-")[0]
        score = estimate(data,predictions(db,rows,key),bootstrap=0)
        recall,precision = score["recall"],score["precision"]
        if recall is None:
            raise ValueError("text recall cannot be estimated from the labeled cells")
        ranking = (recall>=.95, precision or 0, recall) if recall>=.95 else (False,recall,precision or 0)
        if family not in best or ranking>best[family][0]:
            best[family]=(ranking,dict(config=key,model=model,threshold=tau,recall=recall,precision=precision))
    winner = choose_model({TEXT:best["text"][1]["recall"],VL:best["vl"][1]["recall"]})
    return best["dual" if winner==TEXT else "vl"][1], {k:v[1] for k,v in best.items()}


def evidence_for(con,db,row,policy,indexes,llm):
    q = row["q"]
    eligible = {r[0] for r in con.execute("SELECT id FROM turn")}
    if row.get("photo"):
        eligible &= {r[0] for r in con.execute("""SELECT turn_id FROM attachment WHERE path IS NOT NULL
                           AND (mime LIKE 'image/%' OR origin='live_frame')""")}
    terms = list(dict.fromkeys([q,*expand(con,llm,q)]))
    from collections import Counter
    keywords = Counter(t for term in terms for t in matching_turns(con,term) if t in eligible)
    vector = {}
    for idx in indexes:
        hits,_ = idx.recall(q,eligible,policy.get("threshold"),300,evaluation_vector(db,idx.backend,q))
        for tid,v in hits.items():
            vector[tid] = max(v,vector.get(tid,-1))
    ranked,groups = rank_conversations(keywords,vector,con.execute("SELECT conv_id,turn_id FROM turn_conv"))
    material,photos = {},{}
    for _,cid in ranked[:200]:
        result,material[cid] = _previews(con,cid,groups[cid],terms,False,{"judge_prompt_chars_omitted":0})
        photos[cid] = result["images"]
    frame_guard(db,q,material)
    return material,photos


def report(con,db,rows,*,score_only=False,persist=False,llm=None,reranker=None):
    from itertools import batched
    import time
    from .calibration import MeasuredReranker
    data = cells(db,rows)
    selected,comparisons = select_config(db,data,rows)
    policy = {"text_model":selected["model"]}
    if selected["threshold"] is not None:
        policy["threshold"]=selected["threshold"]
    llm = llm or LocalLLM()
    indexes = configured_indexes(con,policy)
    rr = MeasuredReranker(db,reranker or Reranker())
    work = {}; observations = []
    for row in rows:
        q = row["q"]
        material,photos = evidence_for(con,db,row,policy,indexes,llm)
        if list(material)!=predictions(db,[row],selected["config"])[q]:
            raise ValueError("retrieval changed after freezing this pool; refresh and review the frame")
        gold = {cid:y for (query,_,_),(size,labels) in data.items() if query==q for cid,y in labels}
        for chunk in batched(material,16):
            scores = rr.score(q,[material[cid] for cid in chunk])
            observations.extend((v,gold[cid]) for cid,v in zip(chunk,scores,strict=True) if cid in gold)
        work[q]=(material,photos)
    if score_only:
        return dict(queries=len(rows),scores=len(observations),seconds=rr.serving_seconds)
    low,high = cascade_bounds(observations)
    strategies = {"fast":{},"llm":{"judge":"llm"},"cascade":{"judge":"cascade","rerank_low":low,"rerank_high":high}}
    reports={}
    for name,options in strategies.items():
        predicted={}; photohits={}; seconds=[]
        for row in rows:
            q=row["q"]; material,photos=work[q]
            start=time.monotonic(); replay=rr.replayed_seconds
            if name=="fast":
                kept=list(material)
            else:
                judged=_judge(con,llm,q,material,options,rr,use_cache=False)
                kept=[cid for cid in material if judged[cid]["relevant"]]
            seconds.append(time.monotonic()-start+rr.replayed_seconds-replay)
            predicted[q]=set(kept)
            if row.get("photo"):
                target=db.execute("SELECT path FROM target WHERE q=?",(q,)).fetchone()
                photohits[q]=None if not target else any(target[0] in photos[c] for c in kept[:5])
        reports[name]=dict(**breakdown(data,rows,predicted,"fast" if name=="fast" else "strict"),
                           photo_top5=photohits,judgment_seconds=seconds)
    def rank_strategy(name):
        r=reports[name]["text"]
        return r["meets_text_gate"],r["precision"] or 0,r["recall"] or 0,-sum(reports[name]["judgment_seconds"])
    winning=max(("llm","cascade"),key=rank_strategy)
    policy.update(strategies[winning])
    if persist:
        with con:
            con.execute("INSERT INTO meta VALUES('search_policy',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",(json.dumps(policy),))
    return dict(selected=policy,model_comparison=comparisons,reports=reports,review=status(db,rows),bootstrap=1000)
