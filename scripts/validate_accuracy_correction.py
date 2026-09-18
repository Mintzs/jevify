import sys,json,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_formal_benchmark import Runner,question_from_dict,torch
OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918'
if __name__=='__main__':
 data=json.loads((OUT/'fresh-validation.json').read_text())['cases'];r=Runner();r.engine.cuda_graphs=False;r.engine.fused_kernels=();r.engine.shared_attention='off';r.engine.warmup();priors={};counts=collections.defaultdict(collections.Counter)
 with (OUT/'fresh-validation-results.jsonl').open('w') as f:
  for i,c in enumerate(data):
   raw=r.call('ours',c);key=json.dumps(c['questions']);qs={k:question_from_dict(v) for k,v in c['questions'].items()}
   if key not in priors:priors[key]=r.engine.decide('N/A',qs)['answers']['decision']['probabilities']
   ps=raw['answers']['decision']['probabilities'];prior=priors[key];values={k:p/max(prior[k],1e-20) for k,p in ps.items()};total=sum(values.values());corrected={k:p/total for k,p in values.items()}
   hf=r.call('hf',c)
   for method,probs in [('original',ps),('prior_correction',corrected),('hf',hf['answers']['decision']['probabilities'])]:
    ok=max(probs,key=probs.get)==c['expected']['decision'];counts[method][c['group']]+=ok
    f.write(json.dumps({'case_id':c['id'],'group':c['group'],'method':method,'expected':c['expected'],'correct':ok,'probabilities':probs})+'\n');f.flush()
   if (i+1)%20==0:print(i+1,dict(counts),flush=True)
