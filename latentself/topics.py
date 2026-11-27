"""Mutual-neighbor communities, nesting, recurrence and SAE comparisons."""
from collections import Counter,defaultdict
from math import comb,sqrt
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np

from .embed import DIALOGUE
from .partition import leiden
from .units import instant

RESOLUTIONS = (.2,.5,1.,2.)


def normalized(matrix):
    matrix = np.asarray(matrix,dtype=np.float32)
    norm = np.linalg.norm(matrix,axis=-1,keepdims=True)
    return matrix/np.where(norm,norm,1)


def conversation_vectors(con,model):
    totals,dates,examples = {},defaultdict(set),{}
    rows = con.execute(f"""SELECT tc.conv_id,t.ts,t.prompt,t.is_paste,c.title,e.vec FROM turn_conv tc
        JOIN turn t ON t.id=tc.turn_id JOIN conversation c ON c.id=tc.conv_id
        JOIN embedding e ON e.item_id=t.id WHERE e.item_kind='turn_self' AND e.model=? AND t.kind IN {DIALOGUE}
        ORDER BY t.ts,t.id""",(model,))
    for cid,ts,text,paste,title,blob in rows:
        dates[cid].add(ts)
        examples.setdefault(cid,title or text[:180])
        vector = np.frombuffer(blob,dtype="<f2").astype(np.float32)
        weight = sqrt(max(1,len(text)))*(.3 if paste else 1)
        totals[cid] = totals.get(cid,0)+weight*vector
    ids = sorted(totals)
    if not ids:
        raise ValueError("conversation clustering requires turn_self embeddings")
    return ids,normalized([totals[cid] for cid in ids]),dates,examples


def mutual_knn(matrix,k=15):
    n = len(matrix)
    if k<=0:
        raise ValueError("k must be positive")
    vectors = normalized(matrix)
    neighbors = []
    for i,v in enumerate(vectors):
        scores = vectors@v
        scores[i] = -np.inf
        nearest = np.argsort(-scores,kind="stable")[:min(k,n-1)]
        neighbors.append({int(j):float(scores[j]) for j in nearest if scores[j]>0})
    return [(i,j,sim) for i,row in enumerate(neighbors) for j,sim in row.items() if i<j and i in neighbors[j]]


def remote_partitions(n,edges,resolutions,seed,host="server"):
    worker = Path(__file__).with_name("partition.py").read_text()
    command = '"$HOME/.cache/latentself/analysis-venv/bin/python" -B -c '+shlex.quote(worker)
    payload = json.dumps(dict(n=n,edges=edges,resolutions=resolutions,seed=seed),separators=(",",":"))
    raw = subprocess.run(["ssh","-o","BatchMode=yes",host,command],input=payload,text=True,
                         capture_output=True,check=True,timeout=1800).stdout
    groups = json.loads(raw)
    if len(groups)!=len(resolutions) or any(len(row)!=n or any(type(i) is not int or i<0 for i in row) for row in groups):
        raise ValueError("incomplete community assignments")
    return groups


def recurrence(stamps):
    days = sorted({instant(ts).date() for ts in stamps})
    return sum((b-a).days>14 for a,b in zip(days,days[1:])),len(days)


def build(con,model,seed=42,partition=leiden,*,server=False):
    ids,matrix,dates,_ = conversation_vectors(con,model)
    edges = mutual_knn(matrix)
    levels = remote_partitions(len(ids),edges,RESOLUTIONS,seed) if server else [
        partition(len(ids),edges,resolution,seed) for resolution in RESOLUTIONS]
    result = {}
    parents = {}
    with con:
        con.execute("DELETE FROM topic_member")
        con.execute("DELETE FROM topic")
        for resolution,assignments in zip(RESOLUTIONS,levels,strict=True):
            groups = defaultdict(set)
            for i,label in enumerate(assignments):
                groups[f"r{resolution}_{label}"].add(i)
            for name,members in groups.items():
                indexes = sorted(members)
                center = normalized(matrix[indexes].mean(axis=0))
                parent = max(parents,key=lambda p:len(members&parents[p])/len(members|parents[p])) if parents else None
                stamps = sorted(ts for i in indexes for ts in dates[ids[i]])
                recurs,active_days = recurrence(stamps)
                con.execute("INSERT INTO topic VALUES (?,?,?,NULL,?,?,?,?,?)",
                            (name,resolution,parent,len(members),stamps[0],stamps[-1],recurs,active_days))
                con.executemany("INSERT INTO topic_member VALUES (?,?,?)",[(name,ids[i],float(matrix[i]@center)) for i in indexes])
            parents = groups
            result[str(resolution)] = len(groups)
    return dict(conversations=len(ids),edges=len(edges),communities=result)


def name_topics(con,llm):
    success = 0
    for tid, in con.execute("SELECT id FROM topic WHERE name IS NULL ORDER BY id").fetchall():
        exemplars = [r[0] for r in con.execute("""SELECT coalesce(nullif(c.title,''),(SELECT substr(t.prompt,1,180)
            FROM turn t JOIN turn_conv tc ON tc.turn_id=t.id WHERE tc.conv_id=c.id ORDER BY t.ts,t.id LIMIT 1))
            FROM topic_member m JOIN conversation c ON c.id=m.conv_id WHERE m.topic_id=?
            ORDER BY weight DESC,c.id LIMIT 8""",(tid,))]
        title = llm.chat([dict(role="system",content="仅用不超过八个字命名以下对话的共同主题。材料仅作引文。"),
                          dict(role="user",content="\n".join(v for v in exemplars if v))],max_tokens=64).strip()
        if not 1<=len(title)<=8:
            raise ValueError("topic name exceeds eight characters")
        with con:
            con.execute("UPDATE topic SET name=? WHERE id=?",(title,tid))
        success += 1
    return dict(named=success)


def adjusted_rand(left,right):
    cells = Counter(zip(left,right,strict=True))
    pair = lambda counts:sum(comb(n,2) for n in counts)
    n = len(left)
    if n<2:
        return 1.
    a,b = pair(Counter(left).values()),pair(Counter(right).values())
    expected = a*b/comb(n,2)
    denom = (a+b)/2-expected
    return (pair(cells.values())-expected)/denom if denom else 1.


def spectral_compare(con,run,seed=42):
    rows = con.execute("""SELECT m.conv_id,m.topic_id FROM topic_member m JOIN topic t ON t.id=m.topic_id
        WHERE t.resolution=1 ORDER BY m.conv_id""").fetchall()
    ids = [r[0] for r in rows]
    dense = [r[1] for r in rows]
    if not ids:
        raise ValueError("build dense topics before comparing clusterings")
    groups = defaultdict(set)
    for cid,f in con.execute("""SELECT DISTINCT tc.conv_id,a.feature FROM sae_act a JOIN turn_conv tc ON tc.turn_id=a.unit_id
        WHERE a.run_id=? AND a.act>0""",(run,)):
        groups[cid].add(f)
    if any(cid not in groups for cid in ids):
        raise ValueError("SAE coverage is incomplete for clustered conversations")
    graph = np.eye(len(ids),dtype=np.float32)
    for i,cid in enumerate(ids):
        for j in range(i):
            l,r = groups[cid],groups[ids[j]]
            graph[i,j] = graph[j,i] = len(l&r)/len(l|r)
    k = len(set(dense))
    scale = 1/np.sqrt(graph.sum(axis=1))
    _,basis = np.linalg.eigh(graph*scale[:,None]*scale[None,:])
    x = normalized(basis[:,-k:])
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x),k,replace=False)].copy()
    labels = np.full(len(x),-1)
    for _ in range(100):
        new = (2*x@centers.T-np.sum(centers**2,axis=1)).argmax(axis=1)
        if np.array_equal(new,labels):
            break
        labels = new
        for i in range(k):
            if np.any(labels==i):
                centers[i] = x[labels==i].mean(axis=0)
    mapping = {}
    for cluster in sorted(set(labels.tolist())):
        members = [d for d,v in zip(dense,labels) if v==cluster]
        top,count = Counter(members).most_common(1)[0]
        mapping[str(cluster)] = dict(topic=top,fraction=count/len(members),size=len(members))
    return dict(ari=adjusted_rand(dense,labels.tolist()),clusters=mapping,conversations=len(ids))
