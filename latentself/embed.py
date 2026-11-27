"""Build search/profile inputs and incrementally store native-size unit vectors."""
import hashlib
import json
from itertools import batched
from pathlib import Path
from subprocess import SubprocessError
from zipfile import BadZipFile

import numpy as np

from .backends.embed import normalize
from .enrich.images import prepare

DIALOGUE = "('prompt','branch','link_import','live','canvas')"


def turn_inputs(con):
    rows = con.execute(f"""SELECT t.id,t.prompt,a.text,c.title,c.content FROM turn t
        LEFT JOIN answer a ON a.turn_id=t.id AND a.draft_idx=0
        LEFT JOIN canvas c ON c.turn_id=t.id WHERE t.kind IN {DIALOGUE} ORDER BY t.ts,t.id""")
    for row in rows:
        prompt = row["prompt"]
        own = [prompt[:2000]]
        answer = row["text"] or ""
        doc = own.copy()
        if answer:
            doc.append("[回答] " + answer[:500])
        omitted_doc = max(0,len(prompt)-2000) + max(0,len(answer)-500)
        for att in con.execute("SELECT caption,ocr,origin FROM attachment WHERE turn_id=? ORDER BY pos",(row["id"],)):
            caption, ocr = att["caption"] or "", att["ocr"] or ""
            if caption or ocr:
                doc.append("[附件] " + " ".join(filter(None,(caption,ocr[:300]))))
                omitted_doc += max(0,len(ocr)-300)
            if caption and att["origin"] != "generated":
                own.append("[附件] " + caption)
        if row["title"] or row["content"]:
            content = row["content"] or ""
            doc.append("[Canvas] " + (row["title"] or "") + " " + content[:500])
            omitted_doc += max(0,len(content)-500)
        yield row["id"], "\n".join(doc), "\n".join(own), omitted_doc, max(0,len(prompt)-2000)


def image_rows(con):
    return con.execute("""SELECT a.id,a.turn_id,a.path,a.origin,t.prompt FROM attachment a JOIN turn t ON t.id=a.turn_id
        WHERE a.path IS NOT NULL AND (a.mime LIKE 'image/%' OR a.origin='live_frame')
        ORDER BY t.ts,a.pos,a.id""")


def index(con, backend, *, limit=None, include_images=True, turn_ids=None, kinds=("turn","turn_self"), emit=print):
    existing = {(r[0],r[1]):r[2] for r in con.execute(
        "SELECT item_kind,item_id,input_hash FROM embedding WHERE model=?", (backend.model,))}
    stats = dict(success=0, skipped=0, failed=0, failures=[], limited=0, omitted_chars=0, omitted_frames=0)
    def save(batch, encode):
        try:
            vectors = normalize(encode([b[3] for b in batch]))
            if len(vectors) != len(batch):
                raise ValueError("embedding service returned the wrong number of vectors")
            with con:
                con.executemany("""INSERT INTO embedding (item_kind,item_id,model,input_hash,dim,vec)
                    VALUES (?,?,?,?,?,?) ON CONFLICT(item_kind,item_id,model) DO UPDATE SET
                    input_hash=excluded.input_hash,dim=excluded.dim,vec=excluded.vec""",
                    ((kind,key,backend.model,digest,len(vec),vec.astype("<f2").tobytes())
                     for (kind,key,digest,_), vec in zip(batch,vectors,strict=True)))
        except (OSError,ValueError,KeyError) as exc:
            stats["failed"] += len(batch)
            stats["failures"].extend({"kind":kind,"id":key,"error":str(exc)} for kind,key,_,_ in batch)
        else:
            stats["success"] += len(batch)
            emit(f"success {stats['success']}")

    def changed(kind,key,data,encoded):
        digest = hashlib.sha1(encoded).hexdigest()
        if existing.get((kind,key)) == digest:
            stats["skipped"] += 1
            return None
        return kind,key,digest,data

    def pending_texts():
        used = 0
        for tid,doc,own,doc_loss,self_loss in turn_inputs(con):
            if turn_ids is not None and tid not in turn_ids:
                stats["skipped"] += len(kinds)
                continue
            if limit is not None and used >= limit:
                stats["limited"] += len(kinds)
                continue
            pending = []
            for kind,text,loss in (("turn",doc,doc_loss),("turn_self",own,self_loss)):
                if kind in kinds:
                    work = changed(kind,tid,text,text.encode())
                    if work:
                        stats["omitted_chars"] += loss
                        pending.append(work)
            if pending:
                used += 1
                yield from pending
    if kinds:
        for batch in batched(pending_texts(),32):
            save(batch,backend.embed_texts)

    def pending_images():
        used = 0
        for row in image_rows(con):
            if turn_ids is not None and row["turn_id"] not in turn_ids:
                stats["skipped"] += 1
                continue
            if limit is not None and used >= limit:
                stats["limited"] += 1
                continue
            try:
                images,omitted = prepare(Path(row["path"]),row["origin"] == "live_frame")
                if not images:
                    raise ValueError("attachment contains no frames")
                item = (row["prompt"], images)
                encoded = json.dumps(item,separators=(",",":"),ensure_ascii=False).encode()
                work = changed("attachment",row["id"],item,encoded)
                if work:
                    used += 1
                    stats["omitted_frames"] += omitted
                    yield work
            except (OSError,ValueError,SubprocessError,BadZipFile) as exc:
                stats["failed"] += 1
                stats["failures"].append({"kind":"attachment","id":row["id"],"error":str(exc)})
    if include_images and backend.multimodal:
        for batch in batched(pending_images(),4):
            save(batch,backend.embed_images)
    return stats
