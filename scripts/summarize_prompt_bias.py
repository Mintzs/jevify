import json,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/decision-engine/prompt-bias-20260918"
def read(name):
    p=OUT/name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
def correct(r):return r["result"]["status"]=="valid" and all(r["result"]["answers"][k]["choice"]==v for k,v in r["expected"].items())
def metric(rows):
    errors=[];nll=[]
    for r in rows:
        if r["result"]["status"]!="valid":continue
        for key,gold in r["expected"].items():
            ps=r["result"]["answers"][key]["probabilities"]
            errors.append(sum((p-int(k==gold))**2 for k,p in ps.items()));nll.append(-math.log(max(ps[gold],1e-38)))
    return {"n":len(rows),"correct":sum(correct(r) for r in rows),"schema_valid":sum(r["result"]["status"]=="valid" for r in rows),"brier":statistics.mean(errors) if errors else None,"nll":statistics.mean(nll) if nll else None}
def main():
    selection=json.loads((OUT/"selection.json").read_text());candidate=selection["candidate"]
    rows=read("validation.jsonl");speed=read("speed.jsonl");methods=["baseline",candidate]
    summary={"candidate":candidate,"development_qualified":selection["development_qualified"],"validation_rows":len(rows),"suites":{},"core_groups":{},"speed":{}}
    for suite in ("core","additional","robustness","new_holdout"):
        summary["suites"][suite]={m:metric([r for r in rows if r["suite"]==suite and r["method"]==m]) for m in methods}
    summary["held_back_original"]={m:metric([r for r in rows if r["suite"]=="core" and not r["in_development"] and r["method"]==m]) for m in methods}
    for group in sorted({r["group"] for r in rows if r["suite"]=="core"}):
        summary["core_groups"][group]={m:metric([r for r in rows if r["suite"]=="core" and r["group"]==group and r["method"]==m]) for m in methods}
    pairs={}
    for r in rows:pairs.setdefault(r["case_id"],{})[r["method"]]=r
    changes=[]
    for cid,p in pairs.items():
        if len(p)!=2:continue
        a,b=[p[m] for m in methods]
        if correct(a)!=correct(b):changes.append({"case_id":cid,"suite":a["suite"],"group":a["group"],"change":"fixed" if correct(b) else "regressed","expected":a["expected"],"baseline":a["result"]["answers"],"candidate":b["result"]["answers"]})
    summary["core_changes"]={kind:sum(c["suite"]=="core" and c["change"]==kind for c in changes) for kind in ("fixed","regressed")}
    for cid in dict.fromkeys(r["case_id"] for r in speed):
        summary["speed"][cid]={}
        for m in methods:
            rs=[r for r in speed if r["case_id"]==cid and r["method"]==m and r["result"]["status"]=="valid"]
            warm=[r["wall_ms"] for r in rs if r["phase"]=="warm"];first=[r["wall_ms"] for r in rs if r["phase"]=="first_shape"]
            if warm:summary["speed"][cid][m]={"warm_median_ms":statistics.median(warm),"warm_min_ms":min(warm),"warm_max_ms":max(warm),"warm_n":len(warm),"first_shape_median_ms":statistics.median(first)}
    summary["all_valid"]=all(r["result"]["status"]=="valid" for r in rows+speed)
    complete=len(rows)==806
    passes=complete and selection["development_qualified"] and summary["all_valid"]
    if passes:
        passes &= summary["held_back_original"][candidate]["correct"]>summary["held_back_original"]["baseline"]["correct"]
        passes &= all(summary["suites"][s][candidate]["correct"]>=summary["suites"][s]["baseline"]["correct"] for s in ("additional","new_holdout"))
        passes &= all(g[candidate]["correct"]>=g["baseline"]["correct"]-2 for g in summary["core_groups"].values())
    summary["passes_accuracy_promotion_rule"]=bool(passes)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"changed-decisions.json").write_text(json.dumps(changes,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
