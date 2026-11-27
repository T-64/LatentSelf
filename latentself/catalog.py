"""Local candidate catalogs, source provenance and explicit viewing state."""
from collections import defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import unicodedata

from .evidence import content_hash


def create_tables(con):
    for table,columns in {
        "item":"id TEXT PRIMARY KEY,domain TEXT NOT NULL,title TEXT,title_zh TEXT,original_title TEXT,year INTEGER,creators TEXT,countries TEXT,genres TEXT,keywords TEXT,overview TEXT,overview_zh TEXT,runtime INTEGER,imdb_id TEXT,tmdb_id INTEGER,sources TEXT,meta_hash TEXT",
        "item_alias":"item_id TEXT,alias TEXT,PRIMARY KEY(item_id,alias)",
        "item_key":"canonical TEXT PRIMARY KEY,item_id TEXT NOT NULL",
        "item_state":"item_id TEXT PRIMARY KEY,status TEXT,rating INTEGER,source TEXT,evidence TEXT,updated_at TEXT",
        "rec_run":"id TEXT PRIMARY KEY,created_at TEXT,params TEXT,profile_run TEXT",
        "rec_item":"run_id TEXT,item_id TEXT,rank INTEGER,score REAL,score_dense REAL,score_feat REAL,score_recent REAL,reasons TEXT,PRIMARY KEY(run_id,item_id)",
    }.items():
        con.execute(f"CREATE TABLE {table}({columns})")
    con.execute("CREATE UNIQUE INDEX item_imdb ON item(imdb_id) WHERE imdb_id IS NOT NULL")
    con.execute("PRAGMA user_version=5")


def as_list(value):
    if isinstance(value,list):
        return list(dict.fromkeys(str(s).strip() for s in value if str(s).strip()))
    return [s.strip() for s in re.split(r"\s*(?:[、;；]| / )\s*",str(value or "")) if s.strip()]


def dumps(value):
    return json.dumps(value,ensure_ascii=False,separators=(",",":"))


def set_state(con,iid,status,rating,source,evidence=(),when=None):
    priority = dict(chat_mention=1,owner_list=2,feedback=3)
    if status not in ("watched","wanted","dismissed","unknown") or rating not in (-1,0,1,None) or source not in priority:
        raise ValueError("invalid film status")
    old = con.execute("SELECT * FROM item_state WHERE item_id=?",(iid,)).fetchone()
    if old and priority[old["source"]]>priority[source]:
        return False
    seen = list(evidence)
    if old and source==old["source"]:
        seen = list(dict.fromkeys(json.loads(old["evidence"])+seen))
        if old["status"]==status and old["rating"]==rating and seen==json.loads(old["evidence"]):
            return False
    stamp = when or datetime.now(timezone.utc).isoformat()
    con.execute("INSERT OR REPLACE INTO item_state VALUES (?,?,?,?,?,?)",(iid,status,rating,source,dumps(seen),stamp))
    return True


def imdb(value):
    m = re.search(r"\btt\d{5,12}\b",str(value or ""))
    return m[0] if m else None


def import_catalog(con,root,owner_map=None):
    root = Path(root)
    sources = json.loads((root/"tspdt-2026.json").read_text())
    private = json.loads((root/"movies-final-2026.json").read_text())
    owner_map = owner_map if owner_map is not None else json.loads(os.environ.get("LATENTSELF_OWNER_LIST_STATUS","{}"))
    rows = [(src,row) for src in ("recommendations","top1000") for row in sources[src]]
    rows.extend(("owner_list",row) for row in private["records"])
    known = {row["canonicalKey"]:imdb(row.get("imdb")) for _,row in rows if imdb(row.get("imdb"))}
    entries = defaultdict(list)
    for src,row in rows:
        canonical = row["canonicalKey"]
        ext = imdb(row.get("imdb")) or known.get(canonical)
        ident = "film:"+ext if ext else canonical if canonical.startswith("film:") else "film:"+canonical
        entries[ident].append((src,row,ext))
    unknown_owner = 0
    with con:
        for ident,members in entries.items():
            # Existing IDs can already have feedback or vectors. Keep them stable across catalog updates.
            existing = con.execute("SELECT id FROM item WHERE imdb_id=?",(members[0][2],)).fetchone()
            if existing:
                ident = existing[0]
            else:
                for _,row,_ in members:
                    old = con.execute("SELECT item_id FROM item_key WHERE canonical=?",(row["canonicalKey"],)).fetchone()
                    if old:
                        ident = old[0]
                        break
            metadata = {}
            provenance = []
            for src,row,ext in members:
                raw = row.get("raw") or {}
                fields = dict(title=row["title"],original_title=row.get("originalTitle"),title_zh=raw.get("title_zh"),
                    year=row.get("year"),runtime=row.get("minutes"),creators=as_list(row.get("creator")),
                    countries=as_list(row.get("country")),overview_zh=row.get("description"),imdb_id=ext)
                for k,v in fields.items():
                    if v and not metadata.get(k):
                        metadata[k] = v
                entry = dict(src="tspdt-"+src if src!="owner_list" else src)
                if row.get("rank"):
                    entry["rank"] = row["rank"]
                if entry not in provenance:
                    provenance.append(entry)
                if src=="owner_list":
                    status = row.get("status") or owner_map.get(row.get("sourceKey"))
                    if status in ("watched","wanted"):
                        set_state(con,ident,status,None,src)
                    else:
                        unknown_owner += 1
                con.execute("INSERT OR REPLACE INTO item_key VALUES (?,?)",(row["canonicalKey"],ident))
            for name in ("creators","countries"):
                metadata[name] = dumps(metadata.get(name,[]))
            metadata.update(id=ident,domain="film",sources=dumps(provenance))
            metadata["meta_hash"] = content_hash(metadata)
            keys = list(metadata)
            # Preserve TMDB-enriched fields on repeated import.
            updates = ",".join(f"{k}=coalesce(excluded.{k},item.{k})" for k in keys if k not in ("id","domain","sources","meta_hash"))
            con.execute(f"INSERT INTO item({','.join(keys)}) VALUES ({','.join('?' for _ in keys)}) ON CONFLICT(id) "
                        f"DO UPDATE SET {updates},sources=excluded.sources,meta_hash=excluded.meta_hash "
                        "WHERE item.meta_hash IS NOT excluded.meta_hash",list(metadata.values()))
    return dict(imported=len(entries),rows=len(rows),duplicates=len(rows)-len(entries),owner_status_unspecified=unknown_owner)


def normalized(title):
    plain = unicodedata.normalize("NFKC",title).casefold().strip()
    plain = re.sub(r"^(?:the|a|an)\s+","",plain)
    return "".join(c for c in plain if c.isalnum())


def aliases(con):
    total = 0
    with con:
        for row in con.execute("SELECT id,title,title_zh,original_title FROM item"):
            values = {s for s in row[1:] if s}
            values |= {normalized(s) for s in values}
            con.execute("DELETE FROM item_alias WHERE item_id=?",(row[0],))
            con.executemany("INSERT INTO item_alias VALUES (?,?)",[(row[0],a) for a in values if a])
            total += len(values)
    return dict(aliases=total)
