"""Prespecified prompt comparison with a frozen development/validation split."""
import json,time,gc,hashlib,collections,statistics
from pathlib import Path
from run_formal_benchmark import Runner,ROOT,DATA,torch,question_from_dict,existing,append,environment,timing_cases
from prompt_bias_variants import VARIANTS,install,select,mnemonic_codes
OUT=ROOT/"outputs/decision-engine/prompt-bias-20260918"
FRESH=ROOT/"outputs/decision-engine/accuracy-diagnosis-20260918/fresh-validation.json"
def correct(r):return r["result"]["status"]=="valid" and all(r["result"]["answers"][k]["choice"]==v for k,v in r["expected"].items())
def metrics(rows):
    groups={g:{"n":sum(r["group"]==g for r in rows),"correct":sum(correct(r) for r in rows if r["group"]==g)} for g in sorted({r["group"] for r in rows})}
    return {"n":len(rows),"correct":sum(correct(r) for r in rows),"groups":groups,"valid":sum(r["result"]["status"]=="valid" for r in rows)}
def main():
    dev=set(json.loads((OUT/"development-ids.json").read_text()))
    new=json.loads((OUT/"new-holdout.json").read_text())["cases"]
    (OUT/"protocol.json").write_text(json.dumps({"variants":list(VARIANTS),"development_count":len(dev),"selection":"highest development correctness, then smallest group regression; baseline wins ties; candidates must gain at least 3 of 80 and lose at most 1 per development group", "validation":"original 250 (170 held back this iteration), prior additional 80, robustness 15, newly frozen 58; baseline and candidate both evaluated", "promotion":"requires improvement on held-back original cases, no loss on additional 80 or new 58, and no task-group drop greater than 2 cases on original 250; inspect latency before promotion", "probability_adjustment":False,"model_weights_changed":False,"temperature":1.0,"accuracy_graphs":False,"hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/"development-ids.json",OUT/"new-holdout.json",ROOT/"scripts/prompt_bias_variants.py"]}},indent=2),encoding="utf-8")
    r=Runner();e=r.engine;e.cuda_graphs=False;install(e);e.warmup()
    (OUT/"environment.json").write_text(json.dumps(environment(),indent=2),encoding="utf-8")
    done={(x["case_id"],x["method"]) for x in existing(OUT/"development.jsonl")}
    for vi,variant in enumerate(VARIANTS):
        select(e,variant);e.clear_optimization_cache()
        for c in DATA["cases"]:
            if c["id"] not in dev or (c["id"],variant) in done:continue
            row=r.measured("ours",c);row.update(method=variant,expected=c["expected"],group=c["group"])
            append(OUT/"development.jsonl",row)
        rows=[x for x in existing(OUT/"development.jsonl") if x["method"]==variant]
        print("DEVELOPMENT",variant,json.dumps(metrics(rows)),flush=True)
    rows=existing(OUT/"development.jsonl");scores={v:metrics([x for x in rows if x["method"]==v]) for v in VARIANTS}
    base=scores["baseline"]
    def regression(v):return max(base["groups"][g]["correct"]-scores[v]["groups"][g]["correct"] for g in base["groups"])
    ranked=sorted(VARIANTS,key=lambda v:(-scores[v]["correct"],regression(v),VARIANTS.index(v)))
    qualified=[v for v in ranked if v!="baseline" and scores[v]["correct"]>=base["correct"]+3 and regression(v)<=1 and scores[v]["valid"]==80]
    candidate=qualified[0] if qualified else next(v for v in ranked if v!="baseline")
    (OUT/"selection.json").write_text(json.dumps({"scores":scores,"candidate":candidate,"development_qualified":bool(qualified),"ranking":ranked},indent=2),encoding="utf-8")
    print("SELECTED",candidate,"qualified",bool(qualified),flush=True)
    allcases=[("core",c) for c in DATA["cases"]]+[("robustness",c) for c in DATA["robustness_cases"]]+[("additional",c) for c in json.loads(FRESH.read_text())["cases"]]+[("new_holdout",c) for c in new]
    done={(x["case_id"],x["method"]) for x in existing(OUT/"validation.jsonl")}
    for i,(suite,c) in enumerate(allcases):
        for variant in (["baseline",candidate] if i%2==0 else [candidate,"baseline"]):
            if (c["id"],variant) in done:continue
            select(e,variant)
            row=r.measured("ours",c);row.update(method=variant,expected=c["expected"],group=c["group"],suite=suite,in_development=c["id"] in dev)
            append(OUT/"validation.jsonl",row)
        if (i+1)%25==0:print("VALIDATION",i+1,"/",len(allcases),flush=True)
    e.cuda_graphs=True
    speedcases=[next(c for c in DATA["cases"] if c["group"]==g) for g in ("routing","boolean","banking")]+[c for c in timing_cases() if c["id"] in ("speed_4_short","speed_16_long")]
    done={(x["case_id"],x["method"],x["round"],x["repeat"]) for x in existing(OUT/"speed.jsonl")}
    for ci,c in enumerate(speedcases):
        for block in range(2):
            for variant in (["baseline",candidate] if (ci+block)%2==0 else [candidate,"baseline"]):
                if all((c["id"],variant,block,j) in done for j in range(6)):continue
                select(e,variant);e.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
                for j in range(6):
                    row=r.measured("ours",c);row.update(method=variant,round=block,repeat=j,phase="first_shape" if j==0 else "warm")
                    append(OUT/"speed.jsonl",row)
                print("SPEED",c["id"],variant,block,flush=True)
    examples=[]
    for c in speedcases[:3]:
        qs={k:question_from_dict(v) for k,v in c["questions"].items()}
        for variant in ["baseline",candidate]:
            select(e,variant)
            examples.append({"case_id":c["id"],"variant":variant,"prompts":[r.tokenizer.decode(row) for row in e.prepare(c["context"],qs)],"token_ids":{k:e._candidate_tokens(q) for k,q in qs.items()}})
    (OUT/"examples.json").write_text(json.dumps(examples,indent=2),encoding="utf-8")
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
