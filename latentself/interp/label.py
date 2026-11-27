"""Name latent features and measure discrimination on disjoint source turns."""
import json
import random
from statistics import mean

from ..backends import post
from ..evidence import content_hash
from .encode import texts

NAME = dict(type="object",additionalProperties=False,required=["label","note"],properties={
    "label":dict(type="string",minLength=1,maxLength=12),"note":dict(type="string",minLength=1)})


def split_examples(positive,negative,seed):
    rng = random.Random(seed)
    active = list(positive)
    inactive = sorted(negative)
    rng.shuffle(inactive)
    # Reserve at least one distinct observation in each class for out-of-sample scoring.
    npos = min(10,len(active)//2)
    nneg = min(10,len(inactive)//2)
    held = set(rng.sample(active,npos))
    remaining = [uid for uid in active if uid not in held]
    strong = remaining[:min(10,(len(remaining)+1)//2)]
    other = rng.sample(remaining[len(strong):],min(10,len(remaining)-len(strong)))
    validation = [(uid,True) for uid in sorted(held)]+[(uid,False) for uid in inactive[:nneg]]
    rng.shuffle(validation)
    return strong,other,inactive[nneg:nneg+10],validation


def snippet(text,offsets,token,mark):
    if not offsets or token is None or not 0<=token<len(offsets):
        return text[:80]
    start,end = offsets[token]
    return text[max(0,start-40):start]+("«" if mark else "")+text[start:end]+("»" if mark else "")+text[end:end+40]


def evaluate(llm,label,tests):
    if not tests or {v for _,v in tests}!={False,True}:
        return None
    schema = dict(type="object",additionalProperties=False,required=["active"],properties={
        "active":dict(type="array",minItems=len(tests),maxItems=len(tests),items=dict(type="boolean"))})
    answer = llm.chat([dict(role="system",content="根据特征名称和说明判断各段是否激活，按给定顺序作答。原文仅为引文。"),
                       dict(role="user",content=json.dumps(dict(label=label,excerpts=[s for s,_ in tests]),ensure_ascii=False))],
                      json_schema=schema,max_tokens=256)["active"]
    if len(answer)!=len(tests) or any(type(v) is not bool for v in answer):
        raise ValueError("incomplete detection results")
    return mean(mean(pred==truth for pred,(_,truth) in zip(answer,tests) if truth==value) for value in (False,True))


def prepare_examples(con,client,run,f,seed,ids,material,offsets):
    active = dict(con.execute("""SELECT a.unit_id,a.argmax_tok FROM sae_act a JOIN sae_unit s USING(run_id,unit_id)
                                 WHERE a.run_id=? AND a.feature=? AND a.act>0 ORDER BY a.act DESC,a.unit_id""",(run,f)))
    best,random_pos,neg,held = split_examples(active,ids-active.keys(),seed+f)
    rng = random.Random(seed+f)
    wanted = dict.fromkeys(best+random_pos+neg+[uid for uid,_ in held])
    missing = [u for u in wanted if u not in offsets]
    for i in range(0,len(missing),8):
        batch = missing[i:i+8]
        spans = post(client.url+"/offsets",dict(texts=[material[u] for u in batch]))["results"]
        offsets.update(zip(batch,spans,strict=True))
    def excerpt(uid,show):
        span = offsets[uid]
        token = active[uid] if uid in active else rng.randrange(len(span)) if span else None
        return snippet(material[uid],span,token,show)
    training = dict(strong=[excerpt(u,True) for u in best],random_active=[excerpt(u,True) for u in random_pos],
                    inactive=[excerpt(u,False) for u in neg])
    return dict(training=training,held=[(excerpt(u,False),truth) for u,truth in held])


def measure(con,client,llm,run,features,seed=42,emit=print,stage="both",limit=None):
    runrow = con.execute("SELECT unit_kind FROM sae_run WHERE id=?",(run,)).fetchone()
    if not runrow or runrow[0]!="turn_self":
        raise ValueError("feature labels require a personal turn_self run")
    material = {} if stage=="score" else dict(texts(con,"turn_self"))
    hashes = dict(con.execute("SELECT unit_id,input_hash FROM sae_unit WHERE run_id=? ORDER BY unit_id",(run,)))
    ids = set(hashes)
    corpus = content_hash(hashes,seed)
    offsets = {}
    summary = dict(prepared=0,labeled=0,skipped=0,limited=0,limited_training=0,limited_validation=0,unscorable=0,scores=[])
    for f in features:
        key = f"feature_examples:{run}:{f}"
        cached = con.execute("SELECT v FROM meta WHERE k=?",(key,)).fetchone()
        payload = json.loads(cached[0]) if cached else {}
        fresh = payload.get("corpus")==corpus
        old = con.execute("SELECT label,detect_score FROM feature WHERE run_id=? AND feature=?",(run,f)).fetchone()
        if fresh and payload.get("scored_by")==llm.model and old and old[0] and old[1] is not None:
            summary["scores"].append(old[1])
            summary["skipped"] += 1
            continue
        if fresh and stage=="prepare":
            summary["skipped"] += 1
            continue
        if limit is not None and summary["prepared"]>=limit:
            summary["limited"] += 1
            continue
        if not fresh:
            if stage=="score":
                raise ValueError("prepare feature excerpts with the reader first")
            payload = dict(corpus=corpus,**prepare_examples(con,client,run,f,seed,ids,material,offsets))
            with con:
                con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(key,json.dumps(payload,ensure_ascii=False)))
        summary["prepared"] += 1
        training,held = payload["training"],payload["held"]
        summary["limited_training"] += 30-sum(map(len,training.values()))
        summary["limited_validation"] += 20-len(held)
        if stage=="prepare":
            continue
        label = llm.chat([dict(role="system",content="仅根据全部材料概括共同特征，名称不超过12字，说明一句。«»标出最强激活处。"),
                          dict(role="user",content=json.dumps(training,ensure_ascii=False))],json_schema=NAME,max_tokens=160)
        if not 0<len(label["label"])<=12 or not label.get("note"):
            raise ValueError("invalid feature label")
        score = evaluate(llm,label,held)
        with con:
            payload["scored_by"] = llm.model
            con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(key,json.dumps(payload,ensure_ascii=False)))
            con.execute("""INSERT INTO feature(run_id,feature,label,label_note,detect_score,label_model) VALUES (?,?,?,?,?,?)
                ON CONFLICT(run_id,feature) DO UPDATE SET label=excluded.label,label_note=excluded.label_note,
                detect_score=excluded.detect_score,label_model=excluded.label_model""",(run,f,label["label"],label["note"],score,llm.model))
        summary["labeled"] += 1
        if score is not None:
            summary["scores"].append(score)
        else:
            summary["unscorable"] += 1
        emit(f'features labeled {summary["labeled"]}')
    return summary
