import json

from ..analysis.features import enrichment
from ..analysis.report import selected
from ..backends.llm import LocalLLM
from ..exports import export_markdown
from .baseline import import_activations
from .encode import Client,encode,standardize
from .experiment import MODELS,experiment,register,select_saved
from .label import measure


def dispatch(con,out,args):
    client = Client()
    if args.action=="experiment":
        if args.select:
            best,run,metrics = select_saved(con)
            columns = ("model","precision","layer","formula","sample","fvu","delta_ce","dead_fraction","l0",
                       "features_scored","detect_score","seconds","full_hours")
            lines = ["# 读者模型和 SAE 层比较","","| "+" | ".join(columns)+" |","|"+len(columns)*"---|"]
            for group in metrics.values():
                for item in group:
                    lines.append("| "+" | ".join(str(item[k]) for k in columns)+" |")
            lines.extend(["",f'选定 {run}。默认选择 9B。27B 需同时通过检测分数、FVU 和全量时长门槛。'])
            path = export_markdown(con,out/"analysis"/"P3-LAYER-EVAL.md","\n".join(lines))
            return dict(run=run,selected=best,path=str(path))
        if not args.model or not args.precision:
            raise ValueError("supply --model and --precision, or --select after all conditions")
        return experiment(con,client,LocalLLM(),args.model,args.precision,args.count,stage=args.stage)
    run_id = selected(con)
    run = con.execute("SELECT * FROM sae_run WHERE id=?",(run_id,)).fetchone()
    if not run:
        raise ValueError("selected run does not exist")
    key = next(k for k,m in MODELS.items() if m[0]==run["reader"])
    formula = json.loads(run["notes"])["formula"]
    def variant(kind):
        return register(con,key,run["precision"],run["layer"],formula,kind)
    if args.action=="baseline":
        base = variant("baseline")
        result = import_activations(con,client,base)
        result["enrichment"] = enrichment(con,run_id,base)
        return result
    if args.action=="encode":
        rid = variant("turn_answer") if args.answers else run_id
        result = encode(con,client,rid,limit=args.limit)
        if not (result["limited"] or result["failed"] or result.get("interrupted")):
            result["normalization"] = standardize(con,rid)
        return result
    candidates = [r[0] for r in con.execute("""SELECT feature FROM sae_act WHERE run_id=?
        AND act>0 GROUP BY feature HAVING count(*)>=5 ORDER BY feature""",(run_id,))]
    limit = args.limit
    if limit is not None:
        done = {r[0] for r in con.execute("SELECT feature FROM feature WHERE run_id=? AND detect_score IS NOT NULL",(run_id,))}
        candidates = [f for f in candidates if f not in done]
    result = measure(con,client,LocalLLM(),run_id,candidates[:limit],stage=args.stage)
    result["limited"] = max(0,len(candidates)-limit) if limit is not None else 0
    return result
