"""Controlled paired comparison: actual labels versus uncorrected letter IDs."""
import sys,json,time,gc,hashlib,statistics
from pathlib import Path
from run_formal_benchmark import Runner,DATA,ROOT,environment,append,existing,timing_cases,question_from_dict,torch
OUT=ROOT/"outputs/decision-engine/natural-labels-20260918"
OLD=ROOT/"outputs/decision-engine/formal-benchmark-20260918"
FRESH=ROOT/"outputs/decision-engine/accuracy-diagnosis-20260918/fresh-validation.json"

def correct(row):
    r=row["result"]
    return r["status"]=="valid" and all(r["answers"][k]["choice"]==v for k,v in row["expected"].items())

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    old={r["case_id"]:r for r in existing(OLD/"quality.jsonl") if r["method"]=="ours"}
    cases=DATA["cases"]+DATA["robustness_cases"]+json.loads(FRESH.read_text())["cases"]
    core_ids={c["id"] for c in DATA["cases"]}
    failed={cid for cid,r in old.items() if cid in core_ids and not correct(r)}
    cases.sort(key=lambda c:c["id"] not in failed)
    (OUT/"protocol.json").write_text(json.dumps({"failed_original_ids":sorted(failed),"case_count":len(cases),"core_count":len(core_ids),"data_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/"cases.json",FRESH]},"methods":["letters","labels"],"correction":"removed", "temperature":1.0,"quality_timing":"first request shape after model warmup; graph and selected-head caches cleared before each call", "speed_timing":"paired blocks, reversed order on second round; first request then five warmed repeats", "accuracy":"whole case correct; original expected labels unchanged"},indent=2),encoding="utf-8")
    r=Runner();r.engine.warmup()
    (OUT/"environment.json").write_text(json.dumps(environment(),indent=2),encoding="utf-8")
    done={(x["case_id"],x["method"]) for x in existing(OUT/"quality.jsonl")}
    for i,case in enumerate(cases):
        for method in (["letters","labels"] if i%2==0 else ["labels","letters"]):
            if (case["id"],method) in done:continue
            r.engine.answer_encoding=method;r.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
            row=r.measured("ours",case);row.update(method=method,group=case["group"],expected=case["expected"],original_failed=case["id"] in failed,suite="core" if case["id"] in core_ids else "fresh" if case["id"].startswith("fresh_") else "robustness")
            append(OUT/"quality.jsonl",row)
            print(f'quality {i+1}/{len(cases)} {case["id"]} {method} correct={correct(row)} {row["wall_ms"]:.1f}ms {row["result"]["status"]}',flush=True)
    speeds=[c for c in timing_cases() if len(c["questions"]) in (1,4,16) and c["id"].endswith(("short","long"))]
    speeds += [{"id":"speed_multitoken_1","context":"I was charged twice for one order. Please refund the duplicate charge.","questions":{"decision":{"type":"choice","instructions":"Which team should handle this request?","criteria":{"billing support":"Payment errors and duplicate charges","shipping support":"Missing parcels and delivery problems","product returns":"Returning an unwanted physical product"}}}}]
    done={(x["case_id"],x["method"],x["round"],x["repeat"]) for x in existing(OUT/"speed.jsonl")}
    for ci,case in enumerate(speeds):
        for round_ in range(2):
            for method in (["letters","labels"] if (ci+round_)%2==0 else ["labels","letters"]):
                if all((case["id"],method,round_,j) in done for j in range(6)):continue
                r.engine.answer_encoding=method;r.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
                for j in range(6):
                    row=r.measured("ours",case);row.update(method=method,round=round_,repeat=j,phase="first_shape" if j==0 else "warm",fields=len(case["questions"]))
                    append(OUT/"speed.jsonl",row)
                    print(f'speed {case["id"]} {method} round={round_} repeat={j} {row["wall_ms"]:.1f}ms {row["result"]["status"]}',flush=True)
    # Save real prompts, candidate tokenization and unmodified output vectors.
    r.engine.answer_encoding="labels"
    examples=[]
    for case in [DATA["cases"][0],next(c for c in DATA["cases"] if c["group"]=="routing"),speeds[-1]]:
        qs={k:question_from_dict(v) for k,v in case["questions"].items()}
        examples.append({"case":case,"rendered_prompts":[r.tokenizer.decode(x) for x in r.engine.prepare(case["context"],qs)],"label_token_ids":{k:r.engine._candidate_tokens(q) for k,q in qs.items()}})
    (OUT/"examples.json").write_text(json.dumps(examples,indent=2),encoding="utf-8")
    print("COMPLETE",flush=True)

if __name__=="__main__":main()
