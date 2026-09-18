import sys,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import DecisionEngine,MODEL,DATA,question_from_dict,torch
OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'
if __name__=='__main__':
 e=DecisionEngine.from_pretrained(MODEL,device='cuda',local_files_only=True,shared_attention='off');e.warmup();priors={};counts=collections.Counter()
 with (OUT/'prior-full.jsonl').open('w') as f:
  for i,c in enumerate(DATA['cases']):
   qs={k:question_from_dict(v) for k,v in c['questions'].items()};r=e.decide(c['context'],qs);ps=r['answers']['decision']['probabilities'];key=json.dumps(c['questions'])
   if key not in priors:priors[key]=e.decide('N/A',qs)['answers']['decision']['probabilities']
   prior=priors[key];corrected={k:p/max(prior[k],1e-20) for k,p in ps.items()};total=sum(corrected.values());corrected={k:p/total for k,p in corrected.items()}
   ok=max(corrected,key=corrected.get)==c['expected']['decision'];counts[c['group']]+=ok
   f.write(json.dumps({'case_id':c['id'],'variant':'prior_correction','correct':ok,'original':ps,'prior':prior,'probabilities':corrected})+'\n');f.flush()
   if (i+1)%50==0:print(i+1,dict(counts),flush=True)
