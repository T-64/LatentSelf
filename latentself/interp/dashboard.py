"""Generate a self-contained, searchable index and evidence pages on the Mac."""
from collections import Counter
from html import escape
import json
from urllib.parse import quote

from ..analysis.report import search_feature
from ..backends import post
from ..exports import export_markdown
from ..units import instant
from .encode import texts

STYLE = """
:root{color-scheme:light dark;--paper:#f7f9fa;--ink:#152229;--line:#c5d1d5;--accent:#066b67;--tint:#e0f0ec}
@media(prefers-color-scheme:dark){:root{--paper:#142125;--ink:#e4eeef;--line:#3a5257;--accent:#81d3c7;--tint:#234341}}
*{box-sizing:border-box}html{overflow-x:clip}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif}
main{max-width:990px;margin:auto;padding:clamp(20px,5vw,72px) max(20px,env(safe-area-inset-right)) 80px}
h1{font-size:clamp(30px,5vw,58px);line-height:1.14;margin:25px 0}h2{font-size:20px;border-top:2px solid var(--ink);padding-top:14px}
a{color:var(--accent);text-underline-offset:3px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:26px}
section,article{min-width:0}a.row,nav a{display:flex;gap:10px;align-items:center;justify-content:space-between;min-height:44px}
.row{border-bottom:1px solid var(--line);text-decoration:none}.num{font-variant-numeric:tabular-nums;color:var(--ink)}
input{width:100%;min-height:46px;padding:9px 14px;border:1px solid var(--line);border-radius:3px;background:var(--paper);color:var(--ink);font:inherit}
blockquote{margin:0 0 1rem;border-left:3px solid var(--accent);padding:.5rem 1rem;white-space:pre-wrap;overflow-wrap:anywhere}
mark{background:#81dbb8;color:#102c27}mark.peak{outline:2px solid #148665}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}.bar{display:grid;grid-template-columns:8em 1fr 3em;gap:8px}
meter{width:100%;align-self:center;accent-color:var(--accent)}.meta{display:flex;gap:10px 30px;flex-wrap:wrap}.source{display:inline-flex;align-items:center;min-height:44px}
[hidden]{display:none!important}@media(max-width:660px){.grid,.charts{grid-template-columns:1fr;gap:4px}main{padding-top:22px}}
"""


def page(title,body):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width, initial-scale=1, viewport-fit=cover"><title>'+escape(title)+
            '</title><style>'+STYLE+'</style><main>'+body+'</main></html>')


def highlight(text,offsets,acts,peak):
    if len(offsets)!=len(acts) or not 0<peak<len(acts) or acts[peak]!=max(acts):
        raise ValueError("recomputed token peak differs from the stored peak")
    # Several byte-level tokens can map to one character. Render each character once.
    values = {}
    first,last = offsets[peak]
    left,right = max(0,first-80),min(len(text),last+80)
    for (start,end),act in zip(offsets,acts):
        if act>0:
            for char in range(max(left,start),min(right,end)):
                values[char] = max(act,values.get(char,0))
    parts = []
    for i in range(left,right):
        char = escape(text[i])
        if i in values:
            css = ' class="peak"' if first<=i<last else ""
            char = f'<mark{css} title="{values[i]:.4g}">{char}</mark>'
        parts.append(char)
    return "".join(parts),left+len(text)-right


def bars(counts):
    peak = max(counts.values(),default=1) or 1
    return "".join(f'<div class="bar"><span>{escape(str(k))}</span><meter min="0" max="{peak}" value="{v}"></meter>'
                   f'<span class="num">{v}</span></div>' for k,v in counts.items())


def pct(value):
    return "未测量" if value is None else f"{value:.1%}"


def build(con,client,run,out,limit=None):
    spec = con.execute("SELECT layer,notes FROM sae_run WHERE id=?",(run,)).fetchone()
    if not spec:
        raise ValueError("unknown SAE run")
    formula = json.loads(spec["notes"])["formula"]
    features = [dict(r) for r in con.execute("""SELECT f.* FROM feature f WHERE f.run_id=? AND f.label IS NOT NULL
        AND (SELECT count(*) FROM sae_act a JOIN turn t ON t.id=a.unit_id WHERE a.run_id=f.run_id
        AND a.feature=f.feature AND t.is_paste=0)>=5 ORDER BY enrich_z DESC,f.feature""",(run,))]
    material = dict(texts(con,"turn_self"))
    folder = out/"features"
    urls = {}
    omitted_text = 0
    for row in features[:limit]:
        feature = row["feature"]
        hits = search_feature(con,run,feature)
        weeks = Counter(instant(h["ts"]).strftime("%G-W%V") for h in hits)
        maximum = hits[0]["act"]
        hist = Counter(min(9,int(10*h["act"]/maximum)) for h in hits)
        bins = {f"{maximum*i/10:.2g}–{maximum*(i+1)/10:.2g}":hist[i] for i in range(10)}
        score = row["detect_score"]
        title = escape(row["label"])
        body = '<nav><a href="index.html">← 特征索引</a></nav>'+f'<h1>{title}</h1><p>{escape(row["label_note"] or "")}</p>'
        body += (f'<p class="meta"><span>个人频率 {pct(row["freq_self"])}</span><span>基准频率 {pct(row["freq_base"])}</span>'
                 f'<span>检测分数 {pct(score)}</span></p><div class="charts"><section><h2>激活分布</h2>{bars(bins)}</section>'
                 f'<section><h2>每周出现次数</h2>{bars(dict(sorted(weeks.items())))}</section></div><h2>原文</h2>')
        for hit in hits[:20]:
            text = material[hit["turn_id"]]
            reply = post(client.url+"/encode?per_token=1",dict(texts=[text],layers=[spec["layer"]],feature=feature,formula=formula),
                         timeout=600)["results"][0][str(spec["layer"])]
            marked,omitted = highlight(text,reply["offsets"],reply["token_acts"],hit["argmax_tok"])
            omitted_text += omitted
            link = "https://gemini.google.com/app/"+quote(hit["conv_id"] or "",safe="")
            body += f'<article><a class="source" href="{link}">{escape(hit["ts"][:10])} · 出处</a><blockquote>{marked}</blockquote></article>'
        body += f'<p>展示 {min(20,len(hits))} / {len(hits)} 条激活样本</p>'
        co = con.execute("""SELECT b.feature,count(*) AS n FROM sae_act a JOIN sae_act b USING(run_id,unit_id)
            WHERE a.run_id=? AND a.feature=? AND b.feature<>a.feature GROUP BY b.feature ORDER BY n DESC,b.feature LIMIT 10""",
            (run,feature)).fetchall()
        near = post(client.url+"/neighbors",dict(layer=spec["layer"],feature=feature))["results"]
        for heading,neighbors in (("共同激活",co),("解码方向相似度",near)):
            body += f"<h2>{heading}</h2>"+", ".join(f"{int(f)} ({n:.3g})" for f,n in neighbors)
        output = export_markdown(con,folder/f"{feature}.html",page(row["label"],body))
        urls[feature] = output.name
    rankings = (("特有程度","enrich_z",False),("个人频率","freq_self",False),("难以命名","enrich_z",True))
    body = '<h1>特征索引</h1><input type="search" id="filter" aria-label="搜索特征" placeholder="搜索特征名称"><div class="grid">'
    review = {}
    for title,field,uncertain in rankings:
        entries = sorted((r for r in features if r["feature"] in urls and (not uncertain or
            r["detect_score"] is not None and r["detect_score"]<.65)),key=lambda r:-(r[field] or 0))
        review[title] = [str(folder/urls[r["feature"]]) for r in entries[:20 if uncertain else 30]]
        body += f"<section><h2>{title}</h2>"
        for r in entries:
            val = pct(r[field]) if field=="freq_self" else f'{r[field]:.2f}' if r[field] is not None else "未测量"
            body += f'<a class="row" href="{escape(urls[r["feature"]])}"><span>{escape(r["label"])}</span><span class="num">{val}</span></a>'
        body += "</section>"
    body += """</div><script>document.querySelector('#filter').addEventListener('input',e=>{
    const q=e.target.value.trim().toLowerCase();document.querySelectorAll('.row').forEach(row=>{
    row.hidden=!row.firstElementChild.textContent.toLowerCase().includes(q)})})</script>"""
    index = export_markdown(con,folder/"index.html",page("特征索引",body))
    return dict(index=str(index),pages=len(urls),omitted=len(features)-len(urls),omitted_sample_chars=omitted_text,review=review)
