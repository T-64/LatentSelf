"""Fit retrieval thresholds and compare strict strategies on the private owner labels."""
from collections import Counter
from hashlib import sha256
from itertools import batched
import json
from statistics import mean
import time

import numpy as np

from .backends.embed import VL, TEXT, Reranker
from .backends.llm import LocalLLM
from .evaluation import default_backends, evaluation_vector, has_photo_target
from .search import VectorIndex, _judge, _previews, configured_indexes, expand, matching_turns, rank_conversations


def quality(predictions,gold):
    positive = {cid for cid in gold if gold[cid]}
    if not positive:
        raise ValueError("at least one owner-confirmed relevant conversation per query is required")
    hits = positive & predictions
    return {"recall":len(hits)/len(positive),"precision":len(hits)/len(predictions) if predictions else 0.0,
            "unjudged":len(predictions-gold.keys())}


def average(rows):
    return {k:mean(row[k] for row in rows) for k in ("recall","precision")}


def choose_model(recall):
    return TEXT if recall[TEXT]-recall[VL] >= 0.03-1e-12 else VL


def fit_threshold(samples,membership):
    values = np.concatenate([np.fromiter(s["scores"].values(),dtype=float) for s in samples])
    if not values.size:
        raise ValueError("no scores to calibrate")
    thresholds = np.unique(np.quantile(values,np.linspace(0,1,11)))
    best = None
    for threshold in thresholds:
        reports = []
        for s in samples:
            vec = {tid:v for tid,v in s["scores"].items() if tid in s["top"] or v>=threshold}
            ranked,_ = rank_conversations(s["keywords"],vec,membership)
            reports.append(quality({cid for _,cid in ranked[:200]},s["gold"]))
        avg = average(reports)
        rank = (all(r["recall"]>=0.95 for r in reports),avg["recall"],avg["precision"],
                -sum(r["unjudged"] for r in reports),float(threshold))
        if best is None or rank>best[0]:
            best = (rank,float(threshold),reports)
    return best[1:]


def cascade_bounds(observations):
    # Direct decisions are only allowed beyond all observed label conflicts.
    negative = [s for s,relevant in observations if not relevant]
    positive = [s for s,relevant in observations if relevant]
    low = float(np.nextafter(min(positive,default=0.0),-np.inf))
    high = float(np.nextafter(max(negative,default=1.0),np.inf))
    if low>=high:
        midpoint = (low+high)/2
        return midpoint,midpoint
    return low,high


class MeasuredReranker:
    """Replay exact scored batches with their original end-to-end serving cost."""

    def __init__(self, ev, backend):
        self.ev = ev
        self.backend = backend
        self.replayed_seconds = 0.0
        self.serving_seconds = 0.0

    def score(self, query, documents):
        body = json.dumps([query, documents], ensure_ascii=False, separators=(",", ":")).encode()
        key = (self.backend.model, sha256(body).hexdigest())
        row = self.ev.execute("SELECT scores,seconds FROM rerank_batch WHERE model=? AND digest=?", key).fetchone()
        if row:
            scores, seconds = json.loads(row[0]), row[1]
            self.replayed_seconds += seconds
        else:
            start = time.monotonic()
            scores = self.backend.score(query, documents)
            seconds = time.monotonic() - start
            if len(scores) != len(documents) or not np.isfinite(scores).all():
                raise ValueError("reranker returned incomplete or non-finite scores")
            with self.ev:
                self.ev.execute("INSERT OR IGNORE INTO rerank_batch VALUES (?,?,?,?)",
                                (*key, json.dumps(scores), seconds))
        self.serving_seconds += seconds
        return scores


def calibrate(con,ev,rows,backends=None,llm=None,reranker=None,*,score_only=False):
    backends = backends or default_backends()
    llm = llm or LocalLLM()
    reranker = MeasuredReranker(ev, reranker or Reranker())
    membership = [tuple(r) for r in con.execute("SELECT conv_id,turn_id FROM turn_conv")]
    image_turns = {r[0] for r in con.execute("""SELECT turn_id FROM attachment WHERE path IS NOT NULL
                   AND (mime LIKE 'image/%' OR origin='live_frame')""")}
    models,sample_sets = {},{}
    for backend in backends:
        idx = VectorIndex(con,backend,("turn",))
        image_idx = VectorIndex(con,backend,("turn","attachment")) if backend.model==VL and image_turns else idx
        samples,times = [],[]
        for n,row in enumerate(rows,1):
            start = time.monotonic()
            query = row["q"]
            gold = dict(ev.execute("SELECT cid,relevant FROM label WHERE q=?",(query,)))
            pool = {r[0] for r in ev.execute("SELECT cid FROM pool WHERE q=?",(query,))}
            if not pool or pool-gold.keys():
                raise ValueError(f"query {n}: comparison pool requires owner review")
            if row.get("photo") and not has_photo_target(ev,query):
                raise ValueError(f"query {n}: identify the target with --label or --target-photo")
            eligible = image_turns if row.get("photo") else set(idx.ids)
            terms = [query,*expand(con,llm,query)]
            keywords = Counter(tid for t in dict.fromkeys(terms) for tid in matching_turns(con,t) if tid in eligible)
            retrieval = image_idx if row.get("photo") else idx
            scores = retrieval.scores(query,eligible,evaluation_vector(ev,backend,query))
            top = set(sorted(scores,key=lambda tid:(-scores[tid],tid))[:300])
            samples.append(dict(query=query,gold=gold,terms=terms,keywords=keywords,scores=scores,top=top))
            times.append(time.monotonic()-start)
        tau,reports = fit_threshold(samples,membership)
        models[backend.model] = dict(threshold=tau,queries=[{**r,"recall_stage_seconds":t}
                                                            for r,t in zip(reports,times,strict=True)])
        sample_sets[backend.model] = samples
    text_indices = [i for i,r in enumerate(rows) if not r.get("photo")]
    winner = choose_model({m:mean(result["queries"][i]["recall"] for i in text_indices)
                           for m,result in models.items()})
    policy = {"text_model":winner,"threshold":models[winner]["threshold"]}
    work,observations = [],[]
    by_model = {backend.model:backend for backend in backends}
    selected = configured_indexes(con,policy,visual=by_model[VL],text=by_model[TEXT])
    for s in sample_sets[winner]:
        vec = {}
        for idx in selected:
            query_vec = evaluation_vector(ev,idx.backend,s["query"])
            hits,_ = idx.recall(s["query"],s["scores"].keys(),policy["threshold"],300,query_vec)
            for tid,score in hits.items():
                vec[tid] = max(score,vec.get(tid,-1.0))
        ranked,hits = rank_conversations(s["keywords"],vec,membership)
        evidence = {cid:_previews(con,cid,hits[cid],s["terms"],False,{"judge_prompt_chars_omitted":0})[1]
                    for _,cid in ranked[:200]}
        for chunk in batched(evidence,16):
            scores = reranker.score(s["query"],[evidence[cid] for cid in chunk])
            observations.extend((score,s["gold"][cid]) for score,cid in zip(scores,chunk,strict=True) if cid in s["gold"])
        work.append((s,evidence))
    if score_only:
        return {"queries":len(work), "candidates":sum(len(evidence) for _,evidence in work),
                "rerank_serving_seconds":reranker.serving_seconds}
    low,high = cascade_bounds(observations)
    modes = {}
    for strategy in ("llm","cascade"):
        options = dict(judge=strategy)
        if strategy=="cascade":
            options.update(rerank_low=low,rerank_high=high)
        results = []
        for s,evidence in work:
            replay_before = reranker.replayed_seconds
            start = time.monotonic()
            verdicts = _judge(con,llm,s["query"],evidence,options,reranker,use_cache=False)
            result = quality({cid for cid,v in verdicts.items() if v["relevant"]},s["gold"])
            local_seconds = time.monotonic()-start
            replay_seconds = reranker.replayed_seconds-replay_before
            results.append({**result,"seconds":local_seconds+replay_seconds,
                            "replayed_rerank_seconds":replay_seconds})
        modes[strategy] = {"queries":results,"policy":options}
    def strategy_key(name):
        reports = modes[name]["queries"]
        avg = average(reports)
        return (all(r["recall"]>=0.95 and r["precision"]>=0.90 for r in reports),
                avg["precision"],avg["recall"],-sum(r["seconds"] for r in reports))
    policy.update(modes[max(modes,key=strategy_key)]["policy"])
    with con:
        con.execute("INSERT INTO meta VALUES ('search_policy',?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                    (json.dumps(policy),))
    return {"models":models,"strict":modes,"selected":policy}
