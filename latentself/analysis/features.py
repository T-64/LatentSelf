from collections import Counter,defaultdict
from itertools import combinations
from math import erfc,log,sqrt

from ..units import instant


def difference(a,n,b,m):
    if not n or not m:
        raise ValueError("both populations must contain units")
    odds = log((a+.5)/(n-a+.5))-log((b+.5)/(m-b+.5))
    p = (a+b)/(n+m)
    se = sqrt(p*(1-p)*(1/n+1/m))
    z = (a/n-b/m)/se if se else 0.
    return odds,z,erfc(abs(z)/sqrt(2))


def adjust(pvalues):
    ordered = sorted(pvalues,key=pvalues.get,reverse=True)
    result = {}
    cap = 1.
    for i,key in enumerate(ordered):
        cap = min(cap,pvalues[key]*len(ordered)/(len(ordered)-i))
        result[key] = cap
    return result


def unit_features(con,run,eligible=None):
    by_unit = {r[0]:set() for r in con.execute("SELECT unit_id FROM sae_unit WHERE run_id=?",(run,))
               if eligible is None or r[0] in eligible}
    for tid,f in con.execute("SELECT unit_id,feature FROM sae_act WHERE run_id=? AND act>0",(run,)):
        if tid in by_unit:
            by_unit[tid].add(f)
    return by_unit


def counts(units):
    return Counter(f for features in units.values() for f in features)


def enrichment(con,run,baseline):
    eligible = {r[0] for r in con.execute("SELECT id FROM turn WHERE is_paste=0")}
    own = unit_features(con,run,eligible)
    public = unit_features(con,baseline)
    if not own or not public:
        raise ValueError("non-paste personal and public baseline units are required")
    aa,bb = counts(own),counts(public)
    with con:
        for f in aa.keys()|bb.keys():
            odds,z,_ = difference(aa[f],len(own),bb[f],len(public))
            con.execute("""INSERT INTO feature(run_id,feature,freq_self,freq_base,enrich_logodds,enrich_z)
                VALUES (?,?,?,?,?,?) ON CONFLICT(run_id,feature) DO UPDATE SET freq_self=excluded.freq_self,
                freq_base=excluded.freq_base,enrich_logodds=excluded.enrich_logodds,enrich_z=excluded.enrich_z""",
                (run,f,aa[f]/len(own),bb[f]/len(public),odds,z))
    return dict(personal=len(own),baseline=len(public),rankable=sum(n>=5 for n in aa.values()))


def compare(con,run,group_a,group_b):
    if group_a & group_b:
        raise ValueError("comparison groups must be disjoint")
    left,right = unit_features(con,run,group_a),unit_features(con,run,group_b)
    a,b = counts(left),counts(right)
    result = {}
    for f in a.keys()|b.keys():
        odds,z,p = difference(a[f],len(left),b[f],len(right))
        result[f] = dict(feature=f,logodds=odds,z=z,p=p,samples=[])
    q = adjust({f:v["p"] for f,v in result.items()})
    top = sorted(result.values(),key=lambda v:(q[v["feature"]],-abs(v["z"]),v["feature"]))[:30]
    for v in top:
        f = v["feature"]
        v["q"] = q[f]
        group = group_a if v["z"]>=0 else group_b
        v["samples"] = [r[0] for r in con.execute("""SELECT unit_id FROM sae_act WHERE run_id=? AND feature=?
                        ORDER BY act DESC,unit_id""",(run,f)) if r[0] in group][:3]
    return dict(n_a=len(left),n_b=len(right),tested=len(result),features=top)


def npmi(con,run,similarity,min_count=10):
    units = unit_features(con,run)
    individual = counts(units)
    pairs = Counter()
    for features in units.values():
        pairs.update(combinations(sorted(f for f in features if individual[f]>=min_count),2))
    def score(pair,count):
        p = count/len(units)
        if p==1:
            return 0.
        return log(count*len(units)/(individual[pair[0]]*individual[pair[1]]))/-log(p)
    ranked = sorted(((score(pair,c),pair,c) for pair,c in pairs.items() if c>=min_count),
                    key=lambda v:(-v[0],v[1]))
    output = []
    for value,(a,b),c in ranked:
        if similarity(a,b)<=.8:
            output.append(dict(a=a,b=b,npmi=value,count=c))
            if len(output)==50:
                break
    return output


def profile(con,run,half_life_days=180,before=None,limit=50):
    if half_life_days<=0:
        raise ValueError("half life must be positive")
    bound = instant(before) if before else None
    rows = con.execute("""SELECT a.unit_id,a.feature,a.act_z,t.ts,t.is_paste,f.enrich_z FROM sae_act a
        JOIN turn t ON t.id=a.unit_id JOIN feature f ON f.run_id=a.run_id AND f.feature=a.feature
        WHERE t.ts!='' AND a.run_id=? AND f.enrich_z>0 AND a.act_z IS NOT NULL ORDER BY t.ts DESC,t.id""",(run,))
    weights = Counter()
    evidence = defaultdict(list)
    ref = bound
    for uid,feature,act,ts,paste,enrich in rows:
        date = instant(ts)
        if bound and date>=bound:
            continue
        ref = ref or date
        days = (ref-date).total_seconds()/86400
        strength = act*2**(-days/half_life_days)*(1-.7*bool(paste))*enrich
        weights[feature] += strength
        if strength>0:
            source = evidence[feature]
            source.append((strength,uid))
            source.sort(reverse=True)
            del source[3:]
    return [dict(feature=f,weight=w,evidence=[uid for _,uid in evidence[f]]) for f,w in weights.most_common(limit)]
