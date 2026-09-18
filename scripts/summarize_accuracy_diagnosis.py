import json,math,collections,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/accuracy-diagnosis-20260918';BASE=ROOT/'outputs/decision-engine/formal-benchmark-20260918'
def read(name):return [json.loads(x) for x in (OUT/name).read_text().splitlines()]
def quality(rows):
 return {'n':len(rows),'correct':sum(r['correct'] for r in rows),'brier':sum(sum((p-int(k==r['gold']))**2 for k,p in r['p'].items()) for r in rows)/len(rows),'log_loss':sum(-math.log(max(r['p'][r['gold']],1e-12)) for r in rows)/len(rows)}
def main():
 cases=json.loads((BASE/'cases.json').read_text());core={c['id']:c for c in cases['cases']};fresh={c['id']:c for c in json.loads((OUT/'fresh-validation.json').read_text())['cases']};prod=read('production-quality.jsonl');assert len(prod)==345
 baseline=[json.loads(x) for x in (BASE/'quality.jsonl').read_text().splitlines()]
 report={'quality':{},'fresh':{},'groups':{},'speed':{},'schema_valid_production':len(prod)}
 for method in ['ours','hf']:
  rs=[{'id':r['case_id'],'group':r['group'],'p':r['result']['answers']['decision']['probabilities'],'gold':r['expected']['decision'],'correct':r['result']['answers']['decision']['choice']==r['expected']['decision']} for r in baseline if r['method']==method and r['case_id'] in core]
  report['quality'][method]=quality(rs);report['groups'][method]={g:quality([x for x in rs if x['group']==g]) for g in ['banking','routing','boolean','scoring']}
 rs=[{'id':r['case_id'],'group':r['group'],'p':r['result']['answers']['decision']['probabilities'],'gold':r['expected']['decision'],'correct':r['correct']} for r in prod if r['case_id'] in core]
 report['quality']['corrected']=quality(rs);report['groups']['corrected']={g:quality([x for x in rs if x['group']==g]) for g in ['banking','routing','boolean','scoring']}
 for method in ['original','prior_correction','hf']:
  xs=[{'p':r['probabilities'],'gold':r['expected']['decision'],'correct':r['correct']} for r in read('fresh-validation-results.jsonl') if r['method']==method];report['fresh'][method]=quality(xs)
 xs=[{'p':r['result']['answers']['decision']['probabilities'],'gold':r['expected']['decision'],'correct':r['correct']} for r in prod if r['case_id'] in fresh];report['fresh']['production_corrected']=quality(xs)
 parity=read('path-parity.jsonl');by={(r['case_id'],r['mode']):r for r in parity};report['path_parity']={}
 for mode in ['graphs','fused','both']:
  diffs=[];flips=0
  for (cid,m),a in by.items():
   if m!='plain':continue
   b=by[cid,mode];pa=a['result']['answers']['decision']['probabilities'];pb=b['result']['answers']['decision']['probabilities'];diffs.extend(abs(pa[k]-pb[k]) for k in pa);flips+=max(pa,key=pa.get)!=max(pb,key=pb.get)
  report['path_parity'][mode]={'cases':sum(m=='plain' for _,m in by),'max_probability_delta':max(diffs),'choice_flips':flips}
 speed=read('production-speed.jsonl');assert len(speed)==72
 for cid in dict.fromkeys(x['case_id'] for x in speed):
  report['speed'][cid]={}
  for m in ['none','contextual']:
   rows=[r for r in speed if r['case_id']==cid and r['method']==m];warm=rows[1:]
   report['speed'][cid][m]={'first_ms':rows[0]['wall_ms'],'warm_median_ms':statistics.median(r['wall_ms'] for r in warm),'warm_min_ms':min(r['wall_ms'] for r in warm),'warm_max_ms':max(r['wall_ms'] for r in warm),'warm_neutral_forwards':[r['result']['metadata']['answer_bias'].get('neutral_forward_calls',0) for r in warm]}
 # Track fixes and regressions, never just net accuracy.
 old={r['case_id']:r for r in baseline if r['method']=='ours' and r['case_id'] in core};fixed=[];regressed=[];remaining=[]
 for row in prod:
  cid=row['case_id']
  if cid not in core:continue
  before=old[cid]['result']['answers']['decision'];after=row['result']['answers']['decision'];gold=core[cid]['expected']['decision'];new=max(after['probabilities'],key=after['probabilities'].get);was=before['choice']==gold;now=new==gold
  item={'case_id':cid,'context':core[cid]['context'],'expected':gold,'before':before['choice'],'after':new,'probabilities':after['probabilities']}
  if now and not was:fixed.append(item)
  if was and not now:regressed.append(item)
  if not now:remaining.append(item)
 report['changes']={'fixed':len(fixed),'regressed':len(regressed),'remaining':len(remaining)}
 reference={r['case_id']:r for r in read('prior-full.jsonl')}
 deviations=[];flips=0
 for r in prod:
  if r['case_id'] not in reference:continue
  a=r['result']['answers']['decision']['probabilities'];b=reference[r['case_id']]['probabilities']
  deviations.extend(abs(a[k]-b[k]) for k in a)
  flips += max(a,key=a.get)!=max(b,key=b.get)
 report['corrected_plain_vs_optimized']={'cases':250,'max_probability_delta':max(deviations),'choice_flips':flips}

 (OUT/'implementation-summary.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'changed-decisions.json').write_text(json.dumps({'fixed':fixed,'regressed':regressed,'remaining':remaining},indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
