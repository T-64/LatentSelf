"""Evaluate the final fifth of films using only evidence predating the holdout."""
from itertools import product
import json
from math import ceil,log2

from .catalog import current_id,dumps
from .film_state import mention_table
from .recommend import DEFAULT,candidates,ranked_items,taste
from .units import instant

BACKTEST_VERSION = 2


def split(positive):
    ordered = sorted(positive,key=lambda p:(instant(p[1]),p[0]))
    if len(ordered)<2:
        raise ValueError("at least two distinct films with positive dated dialogue evidence are required")
    cutoff = ordered[-ceil(len(ordered)/5)][1]
    training = [(iid,ts) for iid,ts in ordered if instant(ts)<instant(cutoff)]
    held = [iid for iid,ts in ordered if instant(ts)>=instant(cutoff)]
    if not training:
        raise ValueError("the holdout has no earlier training evidence")
    return cutoff,training,held


def historical_states(con,cutoff):
    states = {}
    for iid,ts,status in con.execute("""SELECT m.item_id,t.ts,m.status FROM item_mention m JOIN turn t ON t.id=m.turn_id
        WHERE t.ts!='' AND m.status IN ('watched','wanted') ORDER BY t.ts,m.turn_id"""):
        if instant(ts)<instant(cutoff):
            states[iid] = status
    for ts,value in con.execute("""SELECT t.ts,s.value FROM signal s JOIN turn t ON t.id=s.turn_id
        WHERE s.kind='rec_feedback' ORDER BY t.ts,t.id"""):
        if instant(ts)<instant(cutoff):
            event = json.loads(value)
            states[current_id(con,event["item"])] = {"like":"watched","dislike":"watched","watched":"watched","wanted":"wanted",
                                     "dismiss":"dismissed"}[event["action"]]
    return states


def scores(order,positive):
    gold = set(positive)
    if not gold:
        raise ValueError("empty holdout")
    ranks = []
    for i,iid in enumerate(order,1):
        if iid in gold:
            ranks.append(i)
        if i>=50 and ranks:
            break
    dcg = sum(1/log2(r+1) for r in ranks if r<=20)
    ideal = sum(1/log2(r+1) for r in range(1,min(len(gold),20)+1))
    return dict(recall20=sum(r<=20 for r in ranks)/len(gold),recall50=sum(r<=50 for r in ranks)/len(gold),
                ndcg20=dcg/ideal,mrr=1/min(ranks) if ranks else 0.)


def baseline_scores(pool,held):
    result = {}
    for name,component in (("prior",3),("dense",0)):
        ordered = sorted(pool,key=lambda r:(-r["components"][component],r["id"]))
        result[name] = scores((r["id"] for r in ordered),held)
    return result


def backtest(con,run,model):
    mention_table(con)
    positive = {}
    for iid,ts in con.execute("""SELECT s.item_id,t.ts FROM item_state s JOIN item_mention m ON m.item_id=s.item_id
        JOIN turn t ON t.id=m.turn_id WHERE t.ts!='' AND (s.rating=1 OR s.status='watched') AND (m.status='watched' OR m.sentiment=1)
        AND m.status!='mentioned_only' ORDER BY t.ts,t.id"""):
        positive.setdefault(iid,ts)
    cutoff,training,held = split(positive.items())
    user = taste(con,run,model,before=cutoff)
    state = historical_states(con,cutoff)
    for iid in held:
        state.pop(iid,None)
    pool = candidates(con,run,model,user,state)
    baseline = baseline_scores(pool,held)
    options = [r for r in pool if any(v>0 for _,v in r["contributions"])]
    has_holdout = not set(held).isdisjoint(r["id"] for r in options)
    def evaluate(weights):
        order = (r["id"] for r in ranked_items(options,weights)) if has_holdout else ()
        return scores(order,held)
    params = [DEFAULT]+[tuple(x/5 for x in values) for values in product(range(6),repeat=4)
                         if sum(values)==5 and values[0] and values[1]]
    measure = lambda v:(v[1]["recall50"],v[1]["ndcg20"],v[1]["recall20"],v[1]["mrr"])
    weights,metrics = max(((w,evaluate(w)) for w in params),key=measure)
    passed = all(metrics["recall50"]>value["recall50"] for value in baseline.values())
    outcome = dict(training=len(training),held=len(held),candidates=len(pool),missing_held=len(set(held)-{r["id"] for r in pool}),
                   weights=weights,metrics=metrics,baselines=baseline,grid_size=len(params),passed=passed,
                   model=model,profile_run=run,version=BACKTEST_VERSION,selection="grid evaluated on the chronological holdout")
    with con:
        con.execute("INSERT OR REPLACE INTO meta VALUES ('rec_backtest',?)",(dumps(outcome),))
    return outcome
