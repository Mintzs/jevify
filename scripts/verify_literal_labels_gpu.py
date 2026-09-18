"""Supplement main suite with real-task warm latency and GPU likelihood parity."""
import json,gc,time
from run_formal_benchmark import Runner,ROOT,DATA,torch,question_from_dict,append,existing
OUT=ROOT/"outputs/decision-engine/natural-labels-20260918"

def main():
    r=Runner();e=r.engine;e.warmup()
    failed=set(json.loads((OUT/"protocol.json").read_text())["failed_original_ids"])
    cases=[next(c for c in DATA["cases"] if c["group"]==group and c["id"] in failed) for group in ("banking","routing","boolean")]
    done={(x["case_id"],x["method"],x["round"],x["repeat"]) for x in existing(OUT/"task-speed.jsonl")}
    for ci,case in enumerate(cases):
        for round_ in range(2):
            for method in (["letters","labels"] if (ci+round_)%2==0 else ["labels","letters"]):
                if all((case["id"],method,round_,j) in done for j in range(6)):continue
                e.answer_encoding=method;e.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
                for j in range(6):
                    row=r.measured("ours",case);row.update(method=method,round=round_,repeat=j,phase="first_shape" if j==0 else "warm",fields=1,group=case["group"])
                    append(OUT/"task-speed.jsonl",row)
                print("task speed",case["id"],method,round_,flush=True)
    e.answer_encoding="labels";checks=[]
    for case in cases[:2]:
        e.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
        qs={k:question_from_dict(v) for k,v in case["questions"].items()}
        actual=e.decide(case["context"],qs)
        prompts=e.prepare(case["context"],qs)
        with torch.inference_mode():
            for (key,q),prompt in zip(qs.items(),prompts):
                scores=[]
                for tokens in e._candidate_tokens(q):
                    ids=torch.tensor([prompt+tokens[:-1]],device=e.device)
                    hidden=e.backbone(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[:,len(prompt)-1:]
                    lp=e.head(hidden).float().log_softmax(-1)[0]
                    scores.append(lp.gather(1,torch.tensor(tokens,device=e.device)[:,None]).sum())
                reference=torch.stack(scores).softmax(0).cpu().tolist()
                values=list(actual["answers"][key]["probabilities"].values())
                delta=max(abs(a-b) for a,b in zip(values,reference))
                checks.append({"case_id":case["id"],"field":key,"actual":values,"independent_full_prompt_reference":reference,"max_probability_delta":delta,"passes_one_percentage_point":delta<0.01})
                print("GPU parity",case["id"],delta,flush=True)
    (OUT/"gpu-parity.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if not all(c["passes_one_percentage_point"] for c in checks):raise RuntimeError("GPU sequence scoring differs by more than one percentage point")

if __name__=="__main__":main()
