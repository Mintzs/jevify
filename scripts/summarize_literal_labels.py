"""Summarize paired label comparisons without changing scores or expected answers."""
import json,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/decision-engine/natural-labels-20260918"
OLD=ROOT/"outputs/decision-engine/formal-benchmark-20260918"
def read(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
def ok(r):return r["result"]["status"]=="valid" and all(r["result"]["answers"][k]["choice"]==v for k,v in r["expected"].items())
def metrics(rows):
    valid=[r for r in rows if r["result"]["status"]=="valid"]
    brier=[];nll=[];gold=[]
    for r in valid:
        for key,label in r["expected"].items():
            ps=r["result"]["answers"][key]["probabilities"]
            brier.append(sum((p-int(k==label))**2 for k,p in ps.items()))
            nll.append(-math.log(max(ps[label],1e-38)));gold.append(ps[label])
    return {"count":len(rows),"correct":sum(ok(r) for r in rows),"schema_valid":len(valid),"brier":statistics.mean(brier) if brier else None,"nll":statistics.mean(nll) if nll else None,"mean_gold_probability":statistics.mean(gold) if gold else None}
def main():
    rows=read(OUT/"quality.jsonl"); speeds=read(OUT/"speed.jsonl")+read(OUT/"task-speed.jsonl"); historical=read(OLD/"quality.jsonl")
    summary={"quality_rows":len(rows),"suites":{},"core_groups":{},"speed":{}}
    for suite in ("core","fresh","robustness"):
        summary["suites"][suite]={m:metrics([r for r in rows if r["suite"]==suite and r["method"]==m]) for m in ("letters","labels")}
    for group in sorted({r["group"] for r in rows if r["suite"]=="core"}):
        summary["core_groups"][group]={m:metrics([r for r in rows if r["suite"]=="core" and r["group"]==group and r["method"]==m]) for m in ("letters","labels")}
        summary["core_groups"][group]["historical_hf"]=metrics([r for r in historical if r["group"]==group and r["method"]=="hf" and r["case_id"] in {x["case_id"] for x in rows if x["suite"]=="core"}])
    pairs={}
    for r in rows:pairs.setdefault(r["case_id"],{})[r["method"]]=r
    changes=[]
    for cid,pair in pairs.items():
        if len(pair)!=2:continue
        a,b=pair["letters"],pair["labels"]
        if ok(a)!=ok(b):changes.append({"case_id":cid,"suite":a["suite"],"group":a["group"],"change":"fixed" if ok(b) else "regressed","expected":a["expected"],"letters":a["result"]["answers"],"labels":b["result"]["answers"]})
    summary["core_fixed"]=sum(r["suite"]=="core" and r["change"]=="fixed" for r in changes)
    summary["core_regressed"]=sum(r["suite"]=="core" and r["change"]=="regressed" for r in changes)
    failed=[r for r in rows if r["original_failed"] and r["method"]=="labels"]
    summary["original_failures_retested"]={"count":len(failed),"fixed":sum(ok(r) for r in failed)}
    summary["historical_hf_core"]=metrics([r for r in historical if r["method"]=="hf" and r["case_id"] in {x["case_id"] for x in rows if x["suite"]=="core"}])
    for cid in dict.fromkeys(r["case_id"] for r in speeds):
        summary["speed"][cid]={}
        for m in ("letters","labels"):
            rs=[r for r in speeds if r["case_id"]==cid and r["method"]==m and r["result"]["status"]=="valid"]
            warm=[r["wall_ms"] for r in rs if r["phase"]=="warm"];cold=[r["wall_ms"] for r in rs if r["phase"]=="first_shape"]
            if not warm:continue
            summary["speed"][cid][m]={"warm_median_ms":statistics.median(warm),"warm_min_ms":min(warm),"warm_max_ms":max(warm),"first_shape_median_ms":statistics.median(cold),"warm_samples":len(warm),"peak_allocated_mib":max(r["peak_allocated_mib"] for r in rs)}
    summary["errors"]=[{"case_id":r["case_id"],"method":r["method"],"error":r["result"].get("error")} for r in rows+speeds if r["result"]["status"]!="valid"]
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (OUT/"changed-decisions.json").write_text(json.dumps(changes,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
