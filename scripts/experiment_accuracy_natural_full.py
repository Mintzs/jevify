import sys,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import DecisionEngine,MODEL,DATA,question_from_dict,torch
from experiment_accuracy_answers import make
OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'
if __name__=='__main__':
 e=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,shared_attention='off');e.warmup();counts=collections.Counter()
 with (OUT/'natural-full.jsonl').open('w') as f:
  for i,c in enumerate(DATA['cases']):
   q=question_from_dict(c['questions']['decision']);seq,ids=make(e,c['context'],q,'natural_current')
   with torch.inference_mode():
    h,_=e._hidden([seq]);p=torch.nn.functional.linear(h.float(),e.head.weight[ids].float()).softmax(-1)[0].tolist()
   a=q.answer(p);ok=max(a['probabilities'],key=a['probabilities'].get)==c['expected']['decision'];counts[c['group']]+=ok
   f.write(json.dumps({'case_id':c['id'],'correct':ok,'answer':a})+'\n');f.flush()
   if (i+1)%50==0:print(i+1,dict(counts),flush=True)
