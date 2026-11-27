"""Film representations, historical profiles, ranking and diversity constraints."""
from collections import Counter,defaultdict
from datetime import timedelta
import json
from math import sqrt
import os
import numpy as np

from .analysis.features import difference
from .catalog import current_id
from .evidence import content_hash
from .topics import normalized
from .units import instant

DEFAULT = (.4,.4,.1,.1)


def item_text(row):
    def field(name):
        return "、".join(json.loads(row[name] or "[]"))
    return (f'《{row["title_zh"] or row["title"]}》（{row["original_title"] or ""}，{row["year"] or ""}，{field("countries")}）'
            f'导演：{field("creators")}\n类型：{field("genres")}；关键词：{field("keywords")}\n简介：{row["overview_zh"] or row["overview"] or ""}')


def length_scales(con,run,before=None):
    bound = instant(before) if before else None
    rows = [(r[0],r[1]) for r in con.execute("""SELECT s.unit_id,s.n_tokens,t.ts FROM sae_unit s JOIN turn t ON t.id=s.unit_id
        WHERE s.run_id=? ORDER BY s.n_tokens,s.unit_id""",(run,)) if bound is None or instant(r[2])<bound]
    if not rows:
        raise ValueError("personal SAE coverage is required")
    groups = np.array_split(np.arange(len(rows)),min(10,len(rows)))
    buckets = {rows[int(i)][0]:b for b,g in enumerate(groups) for i in g}
    upper = [rows[int(g[-1])][1] for g in groups]
    moments = defaultdict(lambda:[0.,0.])
    for uid,f,a in con.execute("SELECT unit_id,feature,act FROM sae_act WHERE run_id=?",(run,)):
        if uid in buckets:
            m = moments[buckets[uid],f]
            m[0] += a
            m[1] += a*a
    params = {}
    for (b,f),(total,square) in moments.items():
        n = len(groups[b])
        avg = total/n
        params[b,f] = avg,sqrt(max(0,square/n-avg**2))
    def z(length,feature,activation):
        bucket = min(len(upper)-1,int(np.searchsorted(upper,length)))
        mean,scale = params.get((bucket,feature),(0.,0.))
        return (activation-mean)/scale if scale else 0.
    return z


def representations(con,embed,client,run_id,limit=None):
    spec = con.execute("SELECT * FROM sae_run WHERE id=?",(run_id,)).fetchone()
    z = length_scales(con,run_id)
    con.execute("""CREATE TABLE IF NOT EXISTS item_rep(item_id TEXT,run_id TEXT,input_hash TEXT,n_tokens INTEGER,
        PRIMARY KEY(item_id,run_id))""")
    formula = json.loads(spec["notes"])["formula"]
    result = dict(success=0,skipped=0,limited=0,failed=0,failures=[])
    for row in con.execute("SELECT * FROM item ORDER BY id"):
        iid = row["id"]
        text = item_text(row)
        h = content_hash("film-doc-v1",text)
        dense = con.execute("SELECT input_hash FROM embedding WHERE item_kind='item' AND item_id=? AND model=?",(iid,embed.model)).fetchone()
        sparse = con.execute("SELECT input_hash FROM item_rep WHERE item_id=? AND run_id=?",(iid,run_id)).fetchone()
        if dense and sparse and dense[0]==sparse[0]==h:
            result["skipped"] += 1
            continue
        if limit is not None and result["success"]+result["failed"]>=limit:
            result["limited"] += 1
            continue
        try:
            vec = embed.embed_texts([text],kind="doc")[0] if not dense or dense[0]!=h else None
            act = client.encode([text],[spec["layer"]],formula=formula)[0][str(spec["layer"])] if not sparse or sparse[0]!=h else None
            if act and (act["formula"]!=formula or any(act[k]!=spec[k] for k in ("reader","precision","sae_repo"))):
                raise ValueError("catalog reader does not match the personal run")
            with con:
                if vec is not None:
                    vec = np.asarray(vec,dtype="<f2")
                    if vec.ndim!=1 or not len(vec) or not np.all(np.isfinite(vec)):
                        raise ValueError("invalid item vector")
                    con.execute("INSERT OR REPLACE INTO embedding VALUES ('item',?,?,?,?,?)",
                                (iid,embed.model,len(vec),h,vec.tobytes()))
                if act:
                    uid = "item:"+iid
                    con.execute("DELETE FROM sae_act WHERE run_id=? AND unit_id=?",(run_id,uid))
                    con.executemany("INSERT INTO sae_act VALUES (?,?,?,?,?,?)",[(run_id,uid,f,a,z(act["n_tokens"],f,a),tok)
                                    for f,a,tok in act["features"]])
                    con.execute("INSERT OR REPLACE INTO item_rep VALUES (?,?,?,?)",(iid,run_id,h,act["n_tokens"]))
            result["success"] += 1
        except OSError as exc:
            result["interrupted"] = type(exc).__name__
            break
        except (ValueError,KeyError) as exc:
            result["failed"] += 1
            result["failures"].append(dict(item_id=iid,error=str(exc)))
    return result


def feedback_factors(con,run,before=None):
    latest = {}
    for ts,raw in con.execute("""SELECT t.ts,s.value FROM signal s JOIN turn t ON t.id=s.turn_id
                                 WHERE s.kind='rec_feedback' ORDER BY t.ts,t.id"""):
        if before and instant(ts)>=instant(before):
            continue
        event = json.loads(raw)
        if event["action"] in ("like","dislike"):
            latest[current_id(con,event["item"])] = event["action"]
    factors = defaultdict(lambda:1.)
    for iid,action in latest.items():
        for feature, in con.execute("SELECT feature FROM sae_act WHERE run_id=? AND unit_id=? ORDER BY act DESC,feature LIMIT 10",
                                    (run,"item:"+iid)):
            factors[feature] *= 1.2 if action=="like" else .8
    return {f:min(1.5,max(.5,v)) for f,v in factors.items()}


def taste(con,run,model,before=None,half_life=180):
    items = [(r[0],instant(r[1]),bool(r[2]),np.frombuffer(r[3],dtype="<f2").astype(np.float32))
             for r in con.execute("""SELECT t.id,t.ts,t.is_paste,e.vec FROM turn t JOIN embedding e ON e.item_id=t.id
                 WHERE e.item_kind='turn_self' AND e.model=?""",(model,)) if not before or instant(r[1])<instant(before)]
    if not items:
        raise ValueError("no personal vectors in the date interval")
    ref = instant(before) if before else max(v[1] for v in items)
    allowed = json.loads(os.environ.get("LATENTSELF_CULTURE_TOPICS",'["电影","影视","文化","娱乐","艺术","影评"]'))
    cultural = {tid for tid,name,last in con.execute("""SELECT tc.turn_id,t.name,t.last_ts FROM topic t JOIN topic_member m
        ON m.topic_id=t.id JOIN turn_conv tc ON tc.conv_id=m.conv_id WHERE t.name IS NOT NULL""")
        if any(word in name for word in allowed) and (not before or instant(last)<ref)}
    center = np.zeros_like(items[0][3])
    recent = []
    decays = {}
    nonpaste = set()
    for uid,ts,paste,vec in items:
        age = (ref-ts).total_seconds()/86400
        decay = 2**(-age/half_life)
        decays[uid] = decay*(.3 if paste else 1.)
        if not paste:
            nonpaste.add(uid)
        if uid in cultural:
            center += decay*vec
        if ref-ts<=timedelta(days=30):
            recent.append(vec)
    if np.linalg.norm(center)==0:
        raise ValueError("no eligible culture-topic vectors in the date interval")
    base_row = con.execute("""SELECT b.id FROM sae_run b JOIN sae_run m ON m.reader=b.reader AND m.layer=b.layer
        AND m.sae_repo=b.sae_repo AND m.precision=b.precision AND m.notes=b.notes
        WHERE m.id=? AND b.unit_kind='baseline' LIMIT 1""",(run,)).fetchone()
    if not base_row:
        raise ValueError("a matching public baseline run is required")
    base = base_row[0]
    base_n = con.execute("SELECT count(*) FROM sae_unit WHERE run_id=?",(base,)).fetchone()[0]
    base_counts = dict(con.execute("SELECT feature,count(*) FROM sae_act WHERE run_id=? AND act>0 GROUP BY feature",(base,)))
    lens = dict(con.execute("SELECT unit_id,n_tokens FROM sae_unit WHERE run_id=?",(run,)))
    n = len(nonpaste & lens.keys())
    if not n or not base_n:
        raise ValueError("profile requires encoded non-paste personal and baseline units")
    scale = length_scales(con,run,before)
    weights = Counter()
    counts = Counter()
    evidence = defaultdict(list)
    for uid,f,act in con.execute("SELECT unit_id,feature,act FROM sae_act WHERE run_id=?",(run,)):
        if uid not in decays or uid not in lens:
            continue
        counts[f] += uid in nonpaste
        weights[f] += scale(lens[uid],f,act)*decays[uid]
        source = evidence[f]
        source.append((act,uid))
        source.sort(reverse=True)
        del source[2:]
    adjust = feedback_factors(con,run,before)
    for f in weights:
        enrichment = difference(counts[f],n,base_counts.get(f,0),base_n)[1] if counts[f]>=5 else 0
        weights[f] *= max(0,enrichment)*adjust.get(f,1.)
    top = {f:w for f,w in weights.most_common(200) if w>0}
    if not top:
        raise ValueError("no enriched profile features for this interval")
    return dict(center=normalized(center),recent=normalized(recent) if recent else None,
                weights=top,evidence=evidence,scale=scale)


def prior(sources):
    ranks = [r["rank"] for r in sources if r["src"]=="tspdt-top1000" and r.get("rank")]
    return max(0,min(1,(1000-min(ranks))/999)) if ranks else .3 if any(r["src"]=="tspdt-recommendations" for r in sources) else 0.


def candidates(con,run,model,user,states):
    norm = sqrt(sum(v*v for v in user["weights"].values()))
    results = []
    rows = con.execute("""SELECT i.*,e.vec,r.n_tokens FROM item i JOIN embedding e ON e.item_id=i.id
        JOIN item_rep r ON r.item_id=i.id WHERE e.item_kind='item' AND e.model=? AND r.run_id=? ORDER BY i.id""",(model,run))
    for row in rows:
        iid = row["id"]
        status = states.get(iid,"unknown")
        if status not in ("unknown","wanted"):
            continue
        vec = normalized(np.frombuffer(row["vec"],dtype="<f2").astype(np.float32))
        contrib = [(f,user["weights"][f]*user["scale"](row["n_tokens"],f,a)/norm)
                   for f,a in con.execute("SELECT feature,act FROM sae_act WHERE run_id=? AND unit_id=?",(run,"item:"+iid))
                   if f in user["weights"]]
        dense = float(vec@user["center"])
        recent = float((user["recent"]@vec).max()) if user["recent"] is not None else 0.
        components = [dense,sum(v for _,v in contrib),recent,prior(json.loads(row["sources"]))]
        results.append(dict(id=iid,title=row["title_zh"] or row["title"],components=components,vec=vec,
            creators=json.loads(row["creators"] or "[]"),wanted=status=="wanted",
            contributions=sorted(contrib,key=lambda p:-p[1])[:3]))
    return results


def rank(items,n=20,weights=DEFAULT,diversity=.7):
    if n<1 or len(weights)!=4 or any(w<0 for w in weights) or not np.isclose(sum(weights),1):
        raise ValueError("positive list length and four non-negative weights summing to one are required")
    pool = [dict(row,score=float(np.dot(weights,row["components"])),redundancy=0.) for row in items]
    result = []
    directors = Counter()
    wanted = 0
    for _ in range(n):
        eligible = [r for r in pool if (not r["wanted"] or wanted<4) and not any(directors[c]>=2 for c in r["creators"])]
        if not eligible:
            break
        picked = min(eligible,key=lambda r:(-(diversity*r["score"]-(1-diversity)*r["redundancy"]),r["id"]))
        result.append(picked)
        pool = [r for r in pool if r["id"]!=picked["id"]]
        wanted += picked["wanted"]
        directors.update(picked["creators"])
        for row in pool:
            sim = float(picked["vec"]@row["vec"])
            row["redundancy"] = sim if len(result)==1 else max(row["redundancy"],sim)
    return result
