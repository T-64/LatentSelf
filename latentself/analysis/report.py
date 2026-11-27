"""Local Markdown reports and feature search."""
from datetime import date
import os
import numpy as np

from ..backends.embed import Embedder,VL
from ..exports import export_markdown
from ..search import read_policy
from ..sources import conversation_url
from ..topics import spectral_compare
from . import features,stats


def selected(con):
    run = os.environ.get("LATENTSELF_SAE_RUN") or con.execute("SELECT v FROM meta WHERE k='sae_main_run'").fetchone()
    if not run:
        raise ValueError("select a tested SAE run first")
    return run if isinstance(run,str) else run[0]


def search_feature(con,run,feature):
    rows = con.execute("""SELECT a.unit_id AS turn_id,a.act,a.act_z,a.argmax_tok,t.ts,t.prompt,
        (SELECT min(tc.conv_id) FROM turn_conv tc WHERE tc.turn_id=t.id) AS conv_id FROM sae_act a
        JOIN turn t ON t.id=a.unit_id WHERE a.run_id=? AND a.feature=? ORDER BY a.act DESC,a.unit_id""",(run,feature))
    return [dict(r,url=conversation_url(con,r["conv_id"])) for r in rows]


def unnamed(con,run):
    return [dict(row) for row in con.execute("""SELECT f.feature,f.label,f.detect_score,f.enrich_z FROM feature f
        WHERE run_id=? AND detect_score<.65 AND (SELECT count(*) FROM sae_act a JOIN turn t ON a.unit_id=t.id
        WHERE a.run_id=f.run_id AND a.feature=f.feature AND t.is_paste=0)>=5 ORDER BY enrich_z DESC LIMIT 50""",(run,))]


def label_similarity(con,run):
    labels = dict(con.execute("SELECT feature,label FROM feature WHERE run_id=? AND label IS NOT NULL",(run,)))
    backend = Embedder(read_policy(con).get("text_model",VL))
    vectors = {}
    keys = sorted(set(labels.values()))
    for start in range(0,len(keys),32):
        batch = keys[start:start+32]
        vectors.update(zip(batch,backend.embed_texts(batch),strict=True))
    def similarity(a,b):
        return float(np.dot(vectors[labels[a]],vectors[labels[b]])) if a in labels and b in labels else 0.
    return similarity


def analyze(con,kind,preset=None):
    if kind=="overview":
        return stats.overview(con)
    if kind=="timeline":
        return stats.timeline(con)
    run = selected(con)
    if kind=="diff":
        return features.compare(con,run,*stats.split_groups(con,preset or "night"))
    if kind=="profile":
        return features.profile(con,run)
    if kind=="unnamed":
        return unnamed(con,run)
    if kind=="corr":
        return features.npmi(con,run,label_similarity(con,run))
    if kind=="clusters":
        return spectral_compare(con,run)
    raise ValueError("unknown analysis")


def render(con,value,depth=2):
    def cell(value,key):
        if isinstance(value,float):
            return f"{value:.4g}"
        if isinstance(value,list) and key in ("evidence","samples"):
            parts = []
            for uid in value:
                cid = con.execute("SELECT min(conv_id) FROM turn_conv WHERE turn_id=?",(uid,)).fetchone()[0]
                parts.append(f"[{uid}]({conversation_url(con,cid)})" if cid else str(uid))
            return ", ".join(parts)
        if isinstance(value,dict):
            return ", ".join(f"{k}={v}" for k,v in value.items())
        return str(value).replace("|","\\|").replace("\n"," ")
    if isinstance(value,list) and value and isinstance(value[0],dict):
        keys = list(value[0])
        rows = ["| "+" | ".join(keys)+" |","|"+"---|"*len(keys)]
        rows.extend("| "+" | ".join(cell(item.get(k,""),k) for k in keys)+" |" for item in value)
        return "\n".join(rows)
    if isinstance(value,dict):
        lines = []
        for k,v in value.items():
            if isinstance(v,(dict,list)):
                lines.append("#"*min(depth,6)+" "+k+"\n\n"+render(con,v,depth+1))
            else:
                lines.append(f"- {k}　{cell(v,k)}")
        return "\n\n".join(lines)
    return cell(value,"")


def reports(con,out,kind,preset=None):
    folder = out/"analysis"/date.today().isoformat()
    jobs = [(kind,preset)] if kind!="all" else [(k,None) for k in ("overview","timeline","profile","unnamed","corr","clusters")]
    if kind=="all":
        jobs.extend(("diff",p) for p in ("night","images","long","life"))
    result = dict(files=[],unavailable=[])
    for metric,group in jobs:
        try:
            value = analyze(con,metric,group)
        except ValueError as exc:
            result["unavailable"].append(dict(metric=metric,group=group,reason=str(exc)))
            continue
        name = metric+("-"+group if group else "")
        if group and group not in ("life","night","images","long"):
            name = metric+"-dates"
        markdown = f"# {name}\n\n"+render(con,value)
        result["files"].append(str(export_markdown(con,folder/(name+".md"),markdown)))
    return result
