"""Fetch public movie metadata by numeric TMDB or IMDb identifier."""
import json
import os
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request,urlopen

from .catalog import dumps


class TMDB:
    def __init__(self):
        self.key = os.environ.get("TMDB_API_KEY")
        if not self.key:
            raise ValueError("TMDB_API_KEY is missing")
        self.last = 0.

    def get(self,path,**params):
        if not re.fullmatch(r"/(?:find/tt\d{5,12}|movie/\d+)",path):
            raise ValueError("TMDB requests require a public movie identifier")
        headers = {"Accept":"application/json"}
        if "." in self.key:
            headers["Authorization"] = "Bearer "+self.key
        else:
            params["api_key"] = self.key
        req = Request("https://api.themoviedb.org/3"+path+"?"+urlencode(params),headers=headers)
        for attempt in range(4):
            time.sleep(max(0,.25-(time.monotonic()-self.last)))
            self.last = time.monotonic()
            try:
                with urlopen(req,timeout=20) as response:
                    return json.load(response)
            except HTTPError as e:
                if e.code==404:
                    return {}
                if e.code==429 and attempt<3:
                    time.sleep(max(1,int(e.headers.get("Retry-After","3"))))
                    continue
                raise OSError(f"TMDB HTTP {e.code}") from None


def enrich(con,api,limit=None):
    stats = dict(success=0,cached=0,missing_id=0,not_found=0,limited=0,failed=0,failures=[])
    rows = con.execute("SELECT id,imdb_id,tmdb_id FROM item WHERE domain='film' ORDER BY id").fetchall()
    for iid,imdb,tmdb in rows:
        if not imdb and not tmdb:
            stats["missing_id"] += 1
            continue
        key = "tmdb:v1:"+str(imdb or tmdb)
        saved = con.execute("SELECT v FROM meta WHERE k=?",(key,)).fetchone()
        try:
            if saved:
                data = json.loads(saved[0])
                stats["cached"] += 1
            else:
                if limit is not None and stats["success"]+stats["failed"]+stats["not_found"]>=limit:
                    stats["limited"] += 1
                    continue
                if not tmdb:
                    found = api.get("/find/"+imdb,external_source="imdb_id").get("movie_results",[])
                    tmdb = found[0]["id"] if found else None
                data = {}
                if tmdb:
                    en = api.get("/movie/"+str(int(tmdb)),language="en-US",append_to_response="keywords")
                    zh = api.get("/movie/"+str(int(tmdb)),language="zh-CN")
                    data = dict(tmdb_id=tmdb,title_zh=zh.get("title"),original_title=en.get("original_title"),
                                overview=en.get("overview"),overview_zh=zh.get("overview"),runtime=en.get("runtime"),
                                genres=[r["name"] for r in zh.get("genres",[]) or en.get("genres",[])],
                                keywords=[r["name"] for r in en.get("keywords",{}).get("keywords",[])])
            with con:
                if not saved:
                    con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",(key,dumps(data)))
                if data:
                    con.execute("""UPDATE item SET tmdb_id=?,title_zh=coalesce(nullif(?,''),title_zh),
                        original_title=coalesce(nullif(?,''),original_title),overview=coalesce(nullif(?,''),overview),
                        overview_zh=coalesce(nullif(?,''),overview_zh),runtime=coalesce(?,runtime),genres=?,keywords=? WHERE id=?""",
                        (data["tmdb_id"],data["title_zh"],data["original_title"],data["overview"],data["overview_zh"],
                         data["runtime"],dumps(data["genres"]),dumps(data["keywords"]),iid))
            stats["success" if data else "not_found"] += 1
        except (OSError,KeyError,ValueError) as exc:
            stats["failed"] += 1
            stats["failures"].append(dict(item_id=iid,error=type(exc).__name__))
            if isinstance(exc,OSError):
                break
    stats["with_overview"],stats["total"] = con.execute("""SELECT count(*) FILTER
        (WHERE coalesce(nullif(overview_zh,''),nullif(overview,'')) IS NOT NULL),count(*) FROM item WHERE domain='film'""").fetchone()
    return stats
