"""Checkpoint sparse results and normalize pooled activations by token length."""
from math import isfinite,sqrt
import os

from ..backends import post
from ..embed import DIALOGUE
from ..evidence import content_hash


class Client:
    def __init__(self):
        self.url = os.environ.get("LATENTSELF_READER_URL","http://127.0.0.1:8003").rstrip("/")

    def encode(self,texts,layers,**options):
        return post(self.url+"/encode",dict(texts=texts,layers=layers,**options),timeout=600)["results"]


def texts(con,kind):
    if kind=="turn_answer":
        yield from con.execute(f"""SELECT t.id,a.text FROM turn t JOIN answer a ON a.turn_id=t.id
            WHERE t.kind IN {DIALOGUE} AND a.draft_idx=0 ORDER BY t.id""")
        return
    if kind!="turn_self":
        raise ValueError("unknown unit kind")
    for tid,prompt in con.execute(f"SELECT id,prompt FROM turn WHERE kind IN {DIALOGUE} ORDER BY id"):
        captions = ["[图片] "+r[0] for r in con.execute("""SELECT caption FROM attachment WHERE turn_id=?
            AND origin IS NOT 'generated' AND caption IS NOT NULL AND caption!='' ORDER BY pos""",(tid,))]
        yield tid,"\n".join([prompt,*captions])


def store_unit(con,run_id,uid,digest,data):
    n = int(data["n_tokens"])
    features = data["features"]
    if len(features)>128 or len({f for f,_,_ in features})!=len(features) or any(
        not isinstance(f,int) or f<0 or not isfinite(a) or a<=0 or not 0<p<n for f,a,p in features):
        raise ValueError("invalid sparse activations")
    if not isfinite(data["fvu"]):
        raise ValueError("invalid reconstruction error")
    with con:
        con.execute("DELETE FROM sae_act WHERE run_id=? AND unit_id=?",(run_id,uid))
        con.executemany("INSERT INTO sae_act VALUES (?,?,?,?,NULL,?)",[(run_id,uid,*row) for row in features])
        con.execute("INSERT OR REPLACE INTO sae_unit VALUES (?,?,?,?,?,?)",
                    (run_id,uid,n,data["n_chunks"],data["fvu"],digest))


def encode(con,client,run_id,*,limit=None,emit=print):
    run = con.execute("SELECT * FROM sae_run WHERE id=?",(run_id,)).fetchone()
    if not run:
        raise ValueError("register a selected reader run first")
    stats = dict(success=0,skipped=0,limited=0,failed=0,failures=[])
    for uid,text in texts(con,run["unit_kind"]):
        digest = content_hash(run_id,text)
        if con.execute("SELECT 1 FROM sae_unit WHERE run_id=? AND unit_id=? AND input_hash=?",(run_id,uid,digest)).fetchone():
            stats["skipped"] += 1
            continue
        if limit is not None and stats["success"]+stats["failed"]>=limit:
            stats["limited"] += 1
            continue
        try:
            rows = client.encode([text],[run["layer"]])
            data = rows[0][str(run["layer"])]
            if data["reader"]!=run["reader"] or data["sae_repo"]!=run["sae_repo"] or data["precision"]!=run["precision"]:
                raise ValueError("reader configuration differs from the selected run")
            store_unit(con,run_id,uid,digest,data)
            stats["success"] += 1
        except OSError as exc:
            stats["interrupted"] = type(exc).__name__
            break
        except (KeyError,ValueError,IndexError) as exc:
            stats["failed"] += 1
            stats["failures"].append(dict(unit=uid,error=str(exc)))
        emit(f'encoded {stats["success"]}, failed {stats["failed"]}')
    return stats


def standardize(con,run_id):
    """Compute population moments including implicit zeros, grouped by length decile."""
    with con:
        con.execute("CREATE TEMP TABLE IF NOT EXISTS length_bucket(unit_id TEXT PRIMARY KEY,bucket INTEGER)")
        con.execute("DELETE FROM length_bucket")
        con.execute("""INSERT INTO length_bucket SELECT unit_id,ntile(10) OVER (ORDER BY n_tokens,unit_id)
            FROM sae_unit WHERE run_id=?""",(run_id,))
        counts = dict(con.execute("SELECT bucket,count(*) FROM length_bucket GROUP BY bucket"))
        moments = con.execute("""SELECT b.bucket,a.feature,sum(a.act),sum(a.act*a.act) FROM sae_act a
            JOIN length_bucket b USING(unit_id) WHERE run_id=? GROUP BY b.bucket,a.feature""",(run_id,))
        mean_scale = {}
        for bucket,feature,total,squares in moments:
            mean = total/counts[bucket]
            mean_scale[bucket,feature] = mean,sqrt(max(0,squares/counts[bucket]-mean**2))
        def updates():
            rows = con.execute("""SELECT a.rowid,b.bucket,a.feature,a.act FROM sae_act a
                JOIN length_bucket b USING(unit_id) WHERE run_id=?""",(run_id,))
            for key,bucket,feature,act in rows:
                mean,scale = mean_scale[bucket,feature]
                yield ((act-mean)/scale if scale else 0),key
        con.executemany("UPDATE sae_act SET act_z=? WHERE rowid=?",updates())
    return dict(buckets=len(counts),units=sum(counts.values()))
