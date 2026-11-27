"""lself — command line entry.

  lself ingest [--snapshot DIR]       import a Gemini Takeout snapshot (default: newest)
  lself stats                         what is in the database
  lself grep <text> [--answers] [--images] [-n N]
                                      keyword search, grouped by conversation
  lself show <conv_id> [--full]       a whole conversation in order
  lself open <attachment_id>          open an attachment file
"""
import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from . import config
from .db import connect

DIALOGUE_SQL = "('prompt','branch','link_import','live','canvas')"


def _con():
    return connect(config.DB_PATH)


def cmd_ingest(a):
    from .ingest import gemini
    snap = Path(a.snapshot).expanduser() if a.snapshot else config.latest_gemini_snapshot()
    stats = gemini.ingest(_con(), snap)
    print(f"snapshot {snap}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_stats(a):
    con = _con()
    q = lambda s, *p: con.execute(s, p).fetchall()
    print("snapshot     ", dict(q("SELECT k, v FROM meta")))
    print("turns by kind", dict(q("SELECT kind, COUNT(*) FROM turn GROUP BY kind ORDER BY 2 DESC")))
    print("conversations", q("SELECT COUNT(*) FROM conversation")[0][0])
    print("attachments  ", dict(q("SELECT match, COUNT(*) FROM attachment GROUP BY match")))
    print("time range   ", tuple(q("SELECT MIN(ts), MAX(ts) FROM turn")[0]))


def _matching_turns(con, text, answers):
    from .search import matching_turns
    return matching_turns(con,text,answers)


def cmd_grep(a):
    con = _con()
    ids = _matching_turns(con, a.text, a.answers)
    if not ids:
        print("no match")
        return
    con.execute("CREATE TEMP TABLE hit(turn_id TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO hit VALUES (?)", [(i,) for i in ids])
    img_filter = "AND EXISTS (SELECT 1 FROM attachment x WHERE x.turn_id=t.id)" if a.images else ""
    rows = con.execute(f"""
        SELECT tc.conv_id, c.first_ts, c.last_ts, c.n_turns, COUNT(*) AS hits,
               MIN(t.ts) AS hit_ts, group_concat(t.id) AS turn_ids
        FROM hit h JOIN turn t ON t.id=h.turn_id JOIN turn_conv tc ON tc.turn_id=t.id
        JOIN conversation c ON c.id=tc.conv_id
        WHERE 1=1 {img_filter}
        GROUP BY tc.conv_id ORDER BY hit_ts""").fetchall()
    print(f"{len(rows)} conversations, {len(ids)} matching turns\n")
    for r in rows[-a.n:] if a.n else rows:
        first = con.execute("SELECT prompt FROM turn t JOIN turn_conv tc ON tc.turn_id=t.id WHERE tc.conv_id=? AND t.kind IN ('prompt','branch','link_import','live') ORDER BY t.ts LIMIT 1", (r["conv_id"],)).fetchone()
        print(f"■ {r['first_ts'][:10]} → {r['last_ts'][:10]}  {r['n_turns']}轮  {r['hits']}处命中  {r['conv_id']}")
        print(f"  开头: {_clip(first[0] if first else '', 70)}")
        for tid in r["turn_ids"].split(",")[:3]:
            t = con.execute("SELECT ts, prompt FROM turn WHERE id=?", (tid,)).fetchone()
            print(f"  · {t['ts'][:10]} {_clip(t['prompt'], 90)}")
            if a.images:
                for x in con.execute("SELECT id, orig_name, path FROM attachment WHERE turn_id=? ORDER BY pos", (tid,)):
                    print(f"      [{x['id']}] {x['orig_name'] or ''} → {x['path'] or '(file not found)'}")
        print(f"  https://gemini.google.com/app/{r['conv_id']}\n")


def cmd_show(a):
    con = _con()
    c = con.execute("SELECT * FROM conversation WHERE id=?", (a.conv_id,)).fetchone()
    if not c:
        sys.exit(f"unknown conversation {a.conv_id}")
    print(f"# {a.conv_id}  {c['first_ts'][:10]} → {c['last_ts'][:10]}  {c['n_turns']}轮")
    print(f"https://gemini.google.com/app/{a.conv_id}\n")
    for t in con.execute(f"SELECT t.* FROM turn t JOIN turn_conv tc ON tc.turn_id=t.id WHERE tc.conv_id=? AND t.kind IN {DIALOGUE_SQL} ORDER BY t.ts", (a.conv_id,)):
        print(f"── {t['ts'][:16].replace('T', ' ')} UTC  [{t['kind']}]")
        print("Q:", t["prompt"] if a.full else _clip(t["prompt"], 400))
        for x in con.execute("SELECT id, orig_name, path FROM attachment WHERE turn_id=? ORDER BY pos", (t["id"],)):
            print(f"   📎 [{x['id']}] {x['orig_name'] or ''} → {x['path'] or '(file not found)'}")
        for ans in con.execute("SELECT draft_idx, text FROM answer WHERE turn_id=? ORDER BY draft_idx", (t["id"],)):
            label = "A:" if ans["draft_idx"] == 0 else f"A(草稿{ans['draft_idx'] + 1}):"
            body = ans["text"] if a.full else _clip(ans["text"], 300)
            print(label, textwrap.indent(body, "   ").lstrip())
        cv = con.execute("SELECT title, length(content) n FROM canvas WHERE turn_id=?", (t["id"],)).fetchone()
        if cv:
            print(f"   🖼 Canvas {cv['title'] or ''} ({cv['n'] or 0} chars)")
        print()


def cmd_open(a):
    con = _con()
    r = con.execute("SELECT path FROM attachment WHERE id=?", (a.attachment_id,)).fetchone()
    if not r or not r["path"]:
        sys.exit("attachment not found on disk")
    subprocess.run(["open", r["path"]], check=False)


def _clip(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def cmd_enrich(a):
    from .backends.llm import LocalLLM
    from .enrich import enrich
    result = enrich(_con(), LocalLLM(), do_images=a.images or not a.docs,
                    do_docs=a.docs or not a.images, limit=a.limit, retry_errors=a.retry_errors)
    print(json.dumps(result, ensure_ascii=False))
    if result["failed"]:
        sys.exit(1)


def cmd_embed(a):
    from .backends.embed import Embedder, TEXT, VL
    from .embed import index
    from .evaluation import default_backends
    from .search import read_policy
    con = _con()
    if a.model:
        jobs = [(Embedder(a.model),("turn","turn_self"))]
    elif read_policy(con).get("text_model")==TEXT:
        visual,text = default_backends()
        jobs = [(text,("turn","turn_self")),(visual,())]
    else:
        jobs = [(Embedder(VL),("turn","turn_self"))]
    failed = 0
    for backend,kinds in jobs:
        result = index(con,backend,limit=a.limit,kinds=kinds)
        print(json.dumps({"model":backend.model,**result},ensure_ascii=False))
        failed += result["failed"]
    if failed:
        sys.exit(1)


def cmd_search(a):
    from .search import search
    result = search(_con(),a.query,mode="strict" if a.strict else "fast",images_only=a.images,
                    answers=a.answers,terms=a.terms.split(",") if a.terms is not None else None,
                    since=a.since,until=a.until,sort=a.sort)
    if a.json:
        print(json.dumps(result,ensure_ascii=False))
        return
    for notice in result["notices"]:
        print(notice)
    print(json.dumps(result["counts"],ensure_ascii=False))
    for row in result["results"]:
        print(f"\n{row['first_ts'][:10]} → {row['last_ts'][:10]}  {row['n_turns']}轮  {row['hits']}处命中")
        print(row["first_prompt"])
        for hit in row["snippets"]:
            print(f"  {'+'.join(hit['routes'])} {hit['text']}")
        if row["omitted_snippets"]:
            print(f"  另有 {row['omitted_snippets']} 条命中")
        if a.images:
            print("\n".join(row["images"]))
        if "reason" in row:
            print(row["reason"])
        print(row["url"])


def cmd_eval(a):
    from . import evaluation as ev
    from .backends.llm import LocalLLM
    rows = ev.read_queries(a.queries)
    con = _con()
    labels = ev.evaluation_db(a.queries.parent / "labels.sqlite3")
    if a.index_pool:
        backend = next(b for b in ev.default_backends() if b.model==a.index_pool)
        out = ev.index_pool(con,rows,backend,LocalLLM())
        print(json.dumps(out,ensure_ascii=False))
        if out["failed"]:
            sys.exit(1)
    elif a.prepare:
        for n,out in enumerate(ev.prepare(con,labels,rows,ev.default_backends()),1):
            print(n,json.dumps(out))
    elif a.label:
        print("已标注",ev.label_batch(con,labels,rows))
    elif a.calibrate:
        from .calibration import calibrate
        print(json.dumps(calibrate(con,labels,rows),ensure_ascii=False,indent=2))
    else:
        report = list(ev.evaluate(con,labels,rows))
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if not all(r["accepted"] for r in report):
            sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="lself", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="retrieve conversations by keyword and meaning")
    s.add_argument("query")
    for flag in ("strict","images","answers","json"):
        s.add_argument("--"+flag,action="store_true")
    for flag in ("terms","since","until"):
        s.add_argument("--"+flag)
    s.add_argument("--sort",choices=["time","score"],default="time")
    s.set_defaults(f=cmd_search)
    from .backends.embed import VL, TEXT
    s = sub.add_parser("eval",help="prepare private comparison pools, review labels or measure acceptance")
    s.add_argument("--queries",type=Path,default=config.DATA/"processed/latentself/eval/queries.yaml")
    group = s.add_mutually_exclusive_group()
    group.add_argument("--index-pool",choices=[VL,TEXT])
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--label",action="store_true")
    group.add_argument("--calibrate",action="store_true")
    s.set_defaults(f=cmd_eval)
    s = sub.add_parser("embed", help="index new or changed model inputs")
    s.add_argument("--limit", type=int)
    s.add_argument("--model", choices=[VL,TEXT])
    s.set_defaults(f=cmd_embed)
    s = sub.add_parser("enrich", help="describe images and extract document text")
    s.add_argument("--images", action="store_true")
    s.add_argument("--docs", action="store_true")
    s.add_argument("--limit", type=int)
    s.add_argument("--retry-errors", action="store_true")
    s.set_defaults(f=cmd_enrich)
    s = sub.add_parser("ingest"); s.add_argument("--snapshot"); s.set_defaults(f=cmd_ingest)
    s = sub.add_parser("stats"); s.set_defaults(f=cmd_stats)
    s = sub.add_parser("grep"); s.add_argument("text"); s.add_argument("--answers", action="store_true", help="also search Gemini's answers")
    s.add_argument("--images", action="store_true", help="only turns with attachments"); s.add_argument("-n", type=int, default=0, help="show only the latest N")
    s.set_defaults(f=cmd_grep)
    s = sub.add_parser("show"); s.add_argument("conv_id"); s.add_argument("--full", action="store_true"); s.set_defaults(f=cmd_show)
    s = sub.add_parser("open"); s.add_argument("attachment_id"); s.set_defaults(f=cmd_open)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
