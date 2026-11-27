"""Read-only views for local MCP clients and evidence-backed writing."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Literal

from .evidence import references
from .fact_review import evidence_rows
from .search import search as retrieve
from .sources import conversation_url, turn_order

MAX_CHARS = 20000


def readonly(path):
    db = sqlite3.connect(Path(path).absolute().as_uri()+"?mode=ro",uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def bounded(records,*,limit=None,cap=MAX_CHARS,omitted_chars=0,**metadata):
    """Keep the response valid JSON, preserving source IDs and URLs on partial rows."""
    if limit is not None and limit<0:
        raise ValueError("limit must be non-negative")
    total = len(records)
    result = dict(metadata,results=list(records[:limit]),truncated=False,omitted=total-len(records[:limit]),
                  omitted_chars=omitted_chars)
    def size():
        return len(json.dumps(result,ensure_ascii=False))
    # Find the longest prefix that fits, without serializing the entire archive per dropped row.
    if len(result["results"])>1 and size()>cap:
        low,high = 1,len(result["results"])
        while low<high:
            middle = (low+high+1)//2
            result.update(results=records[:middle],omitted=total-middle)
            if size()<=cap:
                low = middle
            else:
                high = middle-1
        result.update(results=records[:low],omitted=total-low)
    if result["results"] and size()>cap:
        row = deepcopy(result["results"][0])
        result["results"] = [row]
        while (excess:=size()-cap)>0 and row:
            fields = [(len(v),k) for k,v in row.items() if isinstance(v,str)
                      and k not in ("id","turn_id","conv_id","url","period") and v]
            lists = [(len(v),k) for k,v in row.items() if isinstance(v,list) and v]
            if fields:
                length,key = max(fields)
                n = min(length,excess+16)
                row[key] = row[key][:-n]
                result["omitted_chars"] += n
            elif lists:
                row[max(lists)[1]].pop()
                result["omitted"] += 1
            else:
                result["results"].clear()
                result["omitted"] += 1
                break
    result["truncated"] = bool(result["omitted"] or result["omitted_chars"])
    return result


class Reader:
    def __init__(self,path):
        self.path = path

    def search(self,query:str,mode:Literal["fast","strict"]="fast",since:str|None=None,until:str|None=None,
               images_only:bool=False,limit:int|None=None)->dict:
        """Search conversations by text, meaning and optional dates."""
        with closing(readonly(self.path)) as db:
            value = retrieve(db,query,mode=mode,since=since,until=until,images_only=images_only,sort="score")
        return bounded(value.pop("results"),limit=limit,**value)

    def show_conversation(self,conv_id:str,full:bool=False)->dict:
        """Read chronological turns, original questions, answers and local attachment paths."""
        with closing(readonly(self.path)) as db:
            records = []
            omitted = 0
            for r in db.execute(f"""SELECT t.id AS turn_id,t.ts,t.prompt,a.text AS answer FROM turn_conv tc
                JOIN turn t ON t.id=tc.turn_id LEFT JOIN answer a ON a.turn_id=t.id AND a.draft_idx=0
                WHERE tc.conv_id=? ORDER BY {turn_order(db,conv_id)}""",(conv_id,)):
                item = dict(r,conv_id=conv_id,url=conversation_url(db,conv_id))
                if item["answer"] and not full:
                    omitted += max(0,len(item["answer"])-2000)
                    item["answer"] = item["answer"][:2000]
                item["attachments"] = [dict(a) for a in db.execute(
                    "SELECT id,path,mime,caption FROM attachment WHERE turn_id=? ORDER BY pos",(r["turn_id"],))]
                records.append(item)
        return bounded(records,omitted_chars=omitted)

    def timeline(self,since:str,until:str,kinds:list[str]|None=None)->dict:
        """List active conversations in the inclusive date range, with titles and sources."""
        with closing(readonly(self.path)) as db:
            found = db.execute("""SELECT tc.conv_id,c.title,s.summary,s.kinds,min(t.ts) AS first_ts,max(t.ts) AS last_ts
                FROM turn t JOIN turn_conv tc ON tc.turn_id=t.id JOIN conversation c ON c.id=tc.conv_id
                JOIN conv_summary s ON s.conv_id=c.id WHERE t.ts!='' AND substr(t.ts,1,length(?))>=? AND substr(t.ts,1,length(?))<=?
                GROUP BY tc.conv_id ORDER BY first_ts,tc.conv_id""",(since,since,until,until))
            records = []
            for r in found:
                tags = json.loads(r["kinds"])
                if not kinds or set(kinds).intersection(tags):
                    records.append(dict(r,kinds=tags,url=conversation_url(db,r["conv_id"])))
        return bounded(records)

    def digest(self,period:str)->dict:
        """Return a calendar digest and links to the conversations used as evidence."""
        with closing(readonly(self.path)) as db:
            row = db.execute("SELECT period,level,body_md FROM digest WHERE period=?",(period,)).fetchone()
            records = []
            if row:
                sources = []
                for ref in sorted(references(row["body_md"])):
                    kind,key = ref.split(":",1)
                    cid = key if kind=="c" else db.execute("SELECT min(conv_id) FROM turn_conv WHERE turn_id=?",(key,)).fetchone()[0]
                    sources.append(dict(ref=ref,conv_id=cid,url=conversation_url(db,cid)))
                records.append(dict(row,sources=sources))
        return bounded(records)

    def facts(self,query:str|None=None,kind:str|None=None,active_only:bool=True)->dict:
        """Read facts, their verification status and exact source quotations."""
        with closing(readonly(self.path)) as db:
            rows = db.execute("""SELECT * FROM fact WHERE (? IS NULL OR kind=?)
                AND (? IS NULL OR instr(lower(text),lower(?))>0) AND (?=0 OR (status='active' AND verified=1))
                ORDER BY date_from,id""",(kind,kind,query,query,active_only))
            records = []
            for r in rows:
                evidence = [dict(turn_id=e["turn_id"],conv_id=e["conv_id"],quote=e["quote"],url=conversation_url(db,e["conv_id"]))
                            for e in evidence_rows(db,r["id"])]
                records.append(dict(r,evidence=evidence))
        return bounded(records)

    def attachment(self,attachment_id:str)->dict:
        """Return attachment metadata and a local path."""
        with closing(readonly(self.path)) as db:
            rows = db.execute("""SELECT a.id,a.turn_id,tc.conv_id,a.path,a.orig_name,a.mime,a.origin,a.caption,a.ocr,
                a.content_kind,a.tags FROM attachment a LEFT JOIN turn_conv tc ON tc.turn_id=a.turn_id
                WHERE a.id=? ORDER BY tc.conv_id""",(attachment_id,))
            return bounded([dict(r,url=conversation_url(db,r["conv_id"])) for r in rows])

    def stats(self,metric:str)->dict:
        """Report availability of analysis metrics from the interpretation stage."""
        from .analysis.report import analyze
        if metric not in ("overview","timeline","profile","unnamed"):
            return bounded([],available=False,notice="支持 overview、timeline、profile 和 unnamed。")
        with closing(readonly(self.path)) as db:
            try:
                value = analyze(db,metric)
            except ValueError as exc:
                return bounded([],available=False,notice=str(exc))
            return bounded(value if isinstance(value,list) else [value],available=True)
