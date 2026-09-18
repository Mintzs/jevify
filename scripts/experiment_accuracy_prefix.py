import sys,json,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import DecisionEngine,MODEL,DATA,question_from_dict
OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'
if __name__=='__main__':
 chosen=set(json.loads((OUT/'development-case-ids.json').read_text()));e=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,shared_attention='off');e.warmup();original=e.prepare
 with (OUT/'prefix-experiments.jsonl').open('w') as f:
  for prefix in ['Answer:\n','The correct answer is','Based on the context, the answer is']:
   def prep(self,ctx,qs):return [row+self.tokenizer.encode(prefix,add_special_tokens=False) for row in original(ctx,qs)]
   e.prepare=types.MethodType(prep,e);n=0
   for c in DATA['cases']:
    if c['id'] not in chosen:continue
    r=e.decide(c['context'],{k:question_from_dict(v) for k,v in c['questions'].items()});a=r['answers']['decision'];ok=max(a['probabilities'],key=a['probabilities'].get)==c['expected']['decision'];n+=ok
    f.write(json.dumps({'case_id':c['id'],'variant':prefix,'correct':ok,'answer':a})+'\n');f.flush()
   print(repr(prefix),n,len(chosen),flush=True)
