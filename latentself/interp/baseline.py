"""Stream the public, ungated WildChat corpus on the inference server."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory

from ..units import is_paste

DATASET = "allenai/WildChat-1M"


def sample(records,count=15000,seed=42):
    if count<1:
        raise ValueError("sample size must be positive")
    rng = random.Random(seed)
    found = []
    eligible = 0
    for item in records:
        user = next((m for m in item["conversation"] if m.get("role")=="user"),None)
        if not user or str(user.get("language","")).lower() not in ("chinese","zh","zh-cn","zh-tw"):
            continue
        text = user.get("content","")
        if not text.strip() or is_paste(text):
            continue
        uid = "wildchat:"+sha256(item["conversation_hash"].encode()).hexdigest()[:24]
        eligible += 1
        if len(found)<count:
            found.append((uid,text))
        else:
            index = rng.randrange(eligible)
            if index<count:
                found[index] = (uid,text)
    if len(found)!=count:
        raise ValueError(f"only {len(found)} eligible baseline units for {count} requested")
    return sorted(found),eligible


def prepare(path,count,seed,revision=None):
    if sys.platform!="linux":
        raise ValueError("prepare the public corpus on the Linux inference host")
    from huggingface_hub import HfApi,HfFileSystem
    import pyarrow.parquet as pq
    api = HfApi()
    info = api.dataset_info(DATASET,revision=revision)
    if info.gated:
        raise ValueError("dataset requires owner acceptance")
    revision = info.sha
    fs = HfFileSystem()
    def records():
        files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
        for index,name in enumerate(files):
            with fs.open(f"datasets/{DATASET}@{revision}/{name}","rb") as handle:
                parquet = pq.ParquetFile(handle)
                for batch in parquet.iter_batches(batch_size=256,columns=["conversation_hash","conversation"]):
                    yield from batch.to_pylist()
            print(f"read shard {index+1}/{len(files)}",flush=True)
    rows,eligible = sample(records(),count,seed)
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp)/"sample"
        with staged.open("x",encoding="utf-8") as out:
            os.fchmod(out.fileno(),0o600)
            for uid,text in rows:
                out.write(json.dumps(dict(id=uid,text=text,source=DATASET,revision=revision),ensure_ascii=False)+"\n")
        os.link(staged,path)
    return dict(eligible=eligible,sample=len(rows),source=DATASET,revision=revision,seed=seed,license="ODC-BY")


def import_activations(con,client,run_id):
    from .encode import standardize,store_unit
    from ..backends import post
    run = con.execute("SELECT * FROM sae_run WHERE id=?",(run_id,)).fetchone()
    if not run or run["unit_kind"]!="baseline":
        raise ValueError("select a baseline run")
    listing = post(client.url+"/baseline",{"list":True})["results"]
    success = skipped = 0
    for item in listing:
        uid,digest = item["id"],item["hash"]
        if con.execute("SELECT 1 FROM sae_unit WHERE run_id=? AND unit_id=? AND input_hash=?",(run_id,uid,digest)).fetchone():
            skipped += 1
            continue
        formula = json.loads(run["notes"])["formula"]
        data = post(client.url+"/baseline",dict(id=uid,layer=run["layer"],formula=formula),timeout=600)["results"]
        if data["formula"]!=formula or any(data[k]!=run[k] for k in ("reader","sae_repo","precision")):
            raise ValueError("baseline reader differs from selected run")
        store_unit(con,run_id,uid,digest,data)
        with con:
            con.execute("INSERT OR REPLACE INTO baseline_unit VALUES (?,?,?)",(uid,item["source"],data["n_tokens"]))
        success += 1
    return dict(success=success,skipped=skipped,normalization=standardize(con,run_id))


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path",type=Path)
    p.add_argument("--count",type=int,default=15000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--revision")
    a = p.parse_args()
    print(json.dumps(prepare(a.path,a.count,a.seed,a.revision)))
