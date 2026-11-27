"""Local-only P1 review checklist and target-photo picker."""
from base64 import b64decode
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .stratified import check_frame, set_photo


def checklist(db,rows,path):
    check_frame(db,rows)
    blocks = ['<!doctype html><meta charset="utf-8"><title>P1 标注清单</title>',
              '<style>body{max-width:56rem;margin:2rem auto;font:16px/1.6 system-ui}article{border-top:1px solid #ccc}pre{white-space:pre-wrap}</style>',
              '<h1>P1 分层抽样标注清单</h1><p>这是固定样本的只读清单。请在终端用 eval --label 保存 y/n/s，q 可暂停。</p>']
    for pos,q,source,excerpt,url in db.execute("""SELECT s.position,s.q,f.source,f.excerpt,f.url
            FROM sample s JOIN frame f USING(q,cid) ORDER BY s.position"""):
        blocks.append(f'<article><h2>{pos} · {escape(q)}</h2><a href="{escape(url,quote=True)}">{escape(source)} 原对话</a>'
                      f'<pre>{escape(excerpt)}</pre></article>')
    with Path(path).open("x",encoding="utf-8") as f:
        f.write("\n".join(blocks))


def serve_photos(con,db,rows,emit=print):
    from .enrich.images import prepare
    check_frame(db,rows)
    queries={str(i):r["q"] for i,r in enumerate(rows,1) if r.get("photo")}
    if not queries:
        raise ValueError("no photo queries")
    items=con.execute("""SELECT a.path,min(t.source),min(coalesce(a.orig_name,a.stored_name)),max(a.origin='live_frame')
        FROM attachment a JOIN turn t ON t.id=a.turn_id WHERE a.path IS NOT NULL
        AND (a.mime LIKE 'image/%' OR a.origin='live_frame') GROUP BY a.path ORDER BY min(t.ts),a.path""").fetchall()
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def reply(self,body,kind="text/html; charset=utf-8"):
            data=body.encode() if isinstance(body,str) else body
            self.send_response(200)
            self.send_header("Content-Type",kind)
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Referrer-Policy","no-referrer")
            self.send_header("Content-Security-Policy","default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url=urlsplit(self.path)
            args=parse_qs(url.query)
            if url.path.startswith("/"+token+"/img/"):
                idx=url.path.split("/")[-1]
                if not idx.isdecimal() or int(idx)>=len(items):
                    return self.send_error(404)
                path,_,_,archive=items[int(idx)]
                try:
                    pictures,_=prepare(Path(path),bool(archive))
                    if not pictures:
                        return self.send_error(404)
                    return self.reply(b64decode(pictures[0].split(",",1)[1]),"image/jpeg")
                except Exception:
                    return self.send_error(404)
            if url.path!="/"+token+"/":
                return self.send_error(404)
            page=args.get("page",["0"])[0]
            q=args.get("q",[next(iter(queries))])[0]
            if q not in queries or not page.isdecimal():
                return self.send_error(400)
            page=int(page); source=args.get("source",["all"])[0]
            filtered=[(i,r) for i,r in enumerate(items) if source in ("all",r[1])]
            links=[f'<a href="?q={num}">查询 {num} · {escape(txt)}</a>' for num,txt in queries.items()]
            filters=[f'<a href="?q={q}&amp;source={s}">{s}</a>' for s in ("all","gemini","aistudio")]
            body=['<!doctype html><meta charset="utf-8"><title>P1 照片核对</title><style>body{font:16px system-ui}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1em}img{max-width:170px;height:125px;object-fit:contain}button{display:block}small{overflow-wrap:anywhere}</style>',
                  '<h1>选择目标照片</h1>',' · '.join(links),f'<h2>{escape(queries[q])}</h2>',' · '.join(filters)]
            chosen=db.execute("SELECT path FROM target WHERE q=?",(queries[q],)).fetchone()
            if chosen:
                body.append('<p>已保存：'+escape(Path(chosen[0]).name)+'</p>')
            body.append(f'<p>共 {len(filtered)} 个文件。第 {page+1} 页。</p><main>')
            for idx,r in filtered[page*24:page*24+24]:
                body.append(f'<form method="post"><img loading="lazy" src="img/{idx}"><br><small>{escape(r[1])} · {escape(r[2])}</small>'
                            f'<input type="hidden" name="q" value="{q}"><input type="hidden" name="i" value="{idx}">'
                            '<button>选为本查询的目标照片</button></form>')
            body.append('</main>')
            for n,label in ((page-1,"上一页"),(page+1,"下一页")):
                if n>=0 and n*24<len(filtered):
                    body.append(f'<a href="?q={q}&amp;source={escape(source,quote=True)}&amp;page={n}">{label}</a> ')
            self.reply("".join(body))

        def do_POST(self):
            if urlsplit(self.path).path!="/"+token+"/":
                return self.send_error(404)
            size=int(self.headers.get("Content-Length","0"))
            if not 0<size<256:
                return self.send_error(400)
            fields=parse_qs(self.rfile.read(size).decode())
            q=fields.get("q",[""])[0]; idx=fields.get("i",[""])[0]
            if q not in queries or not idx.isdecimal() or int(idx)>=len(items):
                return self.send_error(400)
            set_photo(con,db,rows,int(q),items[int(idx)][0])
            self.send_response(303); self.send_header("Location",f"/{token}/?q={q}");self.end_headers()
    with HTTPServer(("127.0.0.1",0),Handler) as server:
        url=f"http://127.0.0.1:{server.server_port}/{token}/"
        emit(url);webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
