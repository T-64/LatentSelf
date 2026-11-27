"""Resumable matched-reader experiments and aggregate model selection."""
from collections import defaultdict
import json
import random
from statistics import mean
from time import monotonic

from ..evidence import content_hash
from .encode import store_unit,texts
from .label import measure

MODELS = {
    "9b":("Qwen/Qwen3.5-9B-Base","Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",65536,[8,12,16,20,24]),
    "27b":("Qwen/Qwen3.5-27B","Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",81920,[16,24,32,40,48]),
}


def register(con,key,precision,layer,formula,kind,experiment_id=None):
    reader,repo,width,layers = MODELS[key]
    if layer not in layers or (key=="9b" and precision!="bf16") or (key=="27b" and precision not in ("bf16-offload","int8")):
        raise ValueError("unsupported model, layer or precision")
    prefix = f"exp-{experiment_id[:12]}-" if experiment_id else ""
    rid = f"{prefix}{key}-{precision}-L{layer}-{formula}-{kind}"
    with con:
        con.execute("""INSERT OR IGNORE INTO sae_run VALUES (?,?,?,?,?,50,?,?,CURRENT_TIMESTAMP,?)""",
                    (rid,reader,layer,repo,width,precision,kind,json.dumps(dict(formula=formula))))
    return rid


def select(metrics):
    if any(not rows for rows in (metrics.get("9b/bf16"),metrics.get("27b/bf16-offload"),metrics.get("27b/int8"))):
        raise ValueError("all three experimental conditions are required")
    indexed = {(r["model"],r["precision"],r["layer"]):r for rows in metrics.values() for r in rows}
    if any(r["sample"]!=2000 or r["features_scored"]!=50 for r in indexed.values()):
        raise ValueError("selection requires 2,000 matched turns and 50 scored features per layer")
    if any(set(r["layer"] for r in rows)!=set(MODELS[rows[0]["model"]][3]) for rows in metrics.values()):
        raise ValueError("some experimental layers are missing")
    base = max(metrics["9b/bf16"],key=lambda r:(r["detect_score"],-r["fvu"]))
    choices = []
    for r in [*metrics["27b/bf16-offload"],*metrics["27b/int8"]]:
        reference = indexed["27b","bf16-offload",r["layer"]]
        if r["precision"]=="int8" and r["fvu"]>1.1*reference["fvu"]:
            continue
        if r["detect_score"]>=base["detect_score"]+.05 and r["fvu"]<=base["fvu"] and r["full_hours"]<=24:
            choices.append(r)
    return max(choices,key=lambda r:(r["detect_score"],-r["fvu"],-r["full_hours"])) if choices else base


def experiment(con,client,llm,key,precision,count=2000,seed=42,emit=print,stage="both"):
    reader,repo,width,layers = MODELS[key]
    rows = sorted(texts(con,"turn_self"))
    sample = random.Random(seed).sample(rows,min(count,len(rows)))
    if not sample:
        raise ValueError("no personal units")
    sample_key = content_hash([(tid,content_hash(txt)) for tid,txt in sample])
    metrics = []
    for layer in layers:
        prefix = f"sae_exp:{key}:{precision}:{layer}:{sample_key}:"
        def run(uid,text,formula,store=None):
            cache_key = prefix+formula+":"+uid
            saved = con.execute("SELECT v FROM meta WHERE k=?",(cache_key,)).fetchone()
            if saved:
                return json.loads(saved[0])
            if stage=="score":
                raise ValueError("the reader has not finished preparing this matched experiment")
            start = monotonic()
            data = client.encode([text],[layer],formula=formula,experiment=True)[0][str(layer)]
            if any(data[k]!=v for k,v in (("reader",reader),("sae_repo",repo),("precision",precision),("formula",formula))):
                raise ValueError("reader settings differ from this experiment")
            data["seconds"] = monotonic()-start
            if store:
                store_unit(con,store,uid,content_hash(text),data)
            with con:
                con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(cache_key,json.dumps(data,separators=(",",":"))))
            return data
        errors = {}
        for formula in ("card","centered"):
            tmp_run = register(con,key,precision,layer,formula,"turn_self",sample_key)
            pilot = [run(uid,text,formula,tmp_run) for uid,text in sample[:100]]
            errors[formula] = sum(r["sse"] for r in pilot)/max(1e-10,sum(r["variance"] for r in pilot))
        chosen = min(errors,key=errors.get)
        rid = register(con,key,precision,layer,chosen,"turn_self",sample_key)
        active = set()
        total_tokens = ce_tokens = 0
        sse = var = loss = seconds = l0 = 0.
        for uid,text in sample:
            r = run(uid,text,chosen,rid)
            active.update(r["active_features"])
            n = max(0,r["n_tokens"]-1)
            total_tokens += n
            ce_tokens += r["ce_tokens"]
            sse += r["sse"]
            var += r["variance"]
            loss += (r["delta_ce"] or 0)*r["ce_tokens"]
            l0 += n*r["l0"]
            seconds += r["seconds"]
        # Only features with retained source turns and a disjoint test set can be scored.
        candidates = [r[0] for r in con.execute("""SELECT feature FROM sae_act WHERE run_id=? GROUP BY feature
                      HAVING count(*)>=5 AND count(*)<=? ORDER BY feature""",(rid,len(sample)-2))]
        selected = random.Random(seed+layer).sample(candidates,min(50,len(candidates)))
        labels = measure(con,client,llm,rid,selected,seed,emit,stage)
        scores = labels["scores"]
        result = dict(model=key,precision=precision,layer=layer,formula=chosen,pilot_fvu=errors,sample=len(sample),
                      fvu=sse/max(var,1e-10),delta_ce=loss/max(1,ce_tokens),dead_fraction=1-len(active)/width,
                      l0=l0/max(1,total_tokens),detect_score=mean(scores) if scores else None,
                      features_scored=len(scores),seconds=seconds,full_hours=3_000_000*seconds/max(1,total_tokens)/3600,
                      sample_hash=sample_key)
        with con:
            con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(f"sae_metric:{key}:{precision}:{layer}",json.dumps(result)))
        metrics.append(result)
        emit(f"finished {key} {precision} layer {layer}")
    return metrics


def select_saved(con):
    groups = defaultdict(list)
    hashes = set()
    for row in con.execute("SELECT v FROM meta WHERE k LIKE 'sae_metric:%'"):
        item = json.loads(row[0])
        hashes.add(item["sample_hash"])
        groups[item["model"]+"/"+item["precision"]].append(item)
    if len(hashes)!=1:
        raise ValueError("experiment sample hashes differ or no experiments have finished")
    best = select(groups)
    run = register(con,best["model"],best["precision"],best["layer"],best["formula"],"turn_self")
    with con:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('sae_main_run',?)",(run,))
    return best,run,groups
