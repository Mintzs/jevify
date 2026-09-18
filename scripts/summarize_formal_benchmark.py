"""Summarize saved results without modifying outputs or repairing model answers."""
import json,math,statistics,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/formal-benchmark-20260918'
DATA=json.loads((OUT/'cases.json').read_text());core={c['id']:c for c in DATA['cases']}
def read(name):
 p=OUT/name;return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []
def quantile(xs,p):
 xs=sorted(xs)
 if not xs:return None
 pos=(len(xs)-1)*p;lo=int(pos);hi=min(lo+1,len(xs)-1);return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)
def quality(rows):
 valid=[r for r in rows if r['result']['status']=='valid'];correct=0;briers=[];logs=[];cal=[];score_errors=[];readable=readable_correct=0
 for row in rows:
  # A directly readable choice is reported separately from schema compliance.
  answers=row['result'].get('answers')
  if answers is None and 'raw_text' in row['result']:
   try:answers=json.loads(row['result']['raw_text'])
   except (ValueError,TypeError):answers=None
  if isinstance(answers,dict) and set(answers)==set(row['expected']):
   for key,target in row['expected'].items():
    obj=answers.get(key)
    if isinstance(obj,dict) and isinstance(obj.get('choice'),str):
     readable+=1;readable_correct+=obj['choice']==target
  if row['result']['status']!='valid':continue
  for key,target in row['expected'].items():
   answer=row['result']['answers'][key];ps=answer['probabilities'];ok=answer['choice']==target
   correct+=ok;briers.append(sum((v-(label==target))**2 for label,v in ps.items()));logs.append(-math.log(max(ps[target],1e-12)))
   cal.append((ps[answer['choice']],int(ok)))
   if row['group']=='scoring':score_errors.append(abs(sum(int(k)*v for k,v in ps.items())-int(target)))
 bins=[]
 for i in range(5):
  vals=[(p,y) for p,y in cal if i/5<=p and (p<(i+1)/5 or i==4)]
  if vals:bins.append({'low':i/5,'high':(i+1)/5,'n':len(vals),'mean_confidence':statistics.mean(v[0] for v in vals),'accuracy':statistics.mean(v[1] for v in vals)})
 return {'requests':len(rows),'valid':len(valid),'schema_valid_rate':len(valid)/len(rows) if rows else None,'status_counts':dict(collections.Counter(r['result']['status'] for r in rows)),
 'usable_correct':correct,'usable_accuracy':correct/len(rows) if rows else None,'accuracy_on_valid':correct/len(valid) if valid else None,
 'readable_choices':readable,'correct_readable_choices':readable_correct,'readable_choice_accuracy':readable_correct/readable if readable else None,
 'brier_valid_only':statistics.mean(briers) if briers else None,'log_loss_valid_only':statistics.mean(logs) if logs else None,
 'ece_5_bins_valid_only':sum(x['n']*abs(x['accuracy']-x['mean_confidence']) for x in bins)/len(cal) if cal else None,'calibration_bins':bins,
 'score_mae_valid_only':statistics.mean(score_errors) if score_errors else None,
 'attempt_latency_median_ms':statistics.median(r['wall_ms'] for r in rows) if rows else None,'peak_allocated_mib':max((r['peak_allocated_mib'] for r in rows),default=None)}
def main():
 rows=read('quality.jsonl');summary={'complete_core':False,'quality':{},'groups':{},'robustness':{},'cold':{},'speed':{},'label_only':{}}
 for method in ['ours','hf','base']:
  selected=[r for r in rows if r['method']==method and r['case_id'] in core]
  summary['quality'][method]=quality(selected)
  summary['groups'][method]={g:quality([r for r in selected if r['group']==g]) for g in ['banking','routing','boolean','scoring']}
  summary['robustness'][method]=quality([r for r in rows if r['method']==method and r['case_id'] not in core])
 summary['complete_core']=all(v['requests']==250 for v in summary['quality'].values())
 valid_ids=[{r['case_id'] for r in rows if r['method']==m and r['case_id'] in core and r['result']['status']=='valid'} for m in ['ours','hf','base']]
 common=set.intersection(*valid_ids)
 summary['common_valid_quality']={m:quality([r for r in rows if r['method']==m and r['case_id'] in common]) for m in ['ours','hf','base']}
 for method in ['ours','hf','base']:
  selected=[r for r in read('cold.jsonl') if r['method']==method]
  summary['cold'][method]={'n':len(selected),'entry_to_first_answer_seconds':[r['process_entry_to_answer_seconds'] for r in selected],'process_wall_seconds':[r.get('external_process_wall_seconds') for r in selected],'ready_seconds':[r['ready_seconds'] for r in selected],'status_counts':dict(collections.Counter(r['result']['status'] for r in selected))}
 speed=read('speed.jsonl')
 for case in dict.fromkeys(r['case_id'] for r in speed):
  summary['speed'][case]={}
  for method in ['ours','hf','base']:
   allrows=[r for r in speed if r['case_id']==case and r['method']==method];warm=[r for r in allrows if r['phase']=='warm']
   first=[r for r in allrows if r['phase']=='first_shape'];success=[r for r in warm if r['result']['status']=='valid']
   summary['speed'][case][method]={'warm_n':len(warm),'warm_valid':len(success),'first_shape_ms':first[0]['wall_ms'] if first else None,'median_ms':statistics.median(r['wall_ms'] for r in warm) if warm else None,'p95_ms':quantile([r['wall_ms'] for r in warm],.95),'valid_median_ms':statistics.median(r['wall_ms'] for r in success) if success else None,'valid_fields_per_second':sum(r['fields'] for r in success)/(sum(r['wall_ms'] for r in warm)/1000) if warm else None,'peak_allocated_mib':max((r['peak_allocated_mib'] for r in allrows),default=None),'failures':[{'status':r['result']['status'],'error':r['result'].get('error')} for r in allrows if r['result']['status']!='valid']}
 labels=read('labels.jsonl')
 if labels:
  valid=[r for r in labels if r['result']['status']=='valid'];correct=sum(r['result']['labels'].get(k)==v for r in valid for k,v in r['expected'].items())
  summary['label_only']={'n':len(labels),'valid':len(valid),'usable_correct':correct,'median_ms':statistics.median(r['wall_ms'] for r in labels),
   'groups':{g:{'n':sum(r['group']==g for r in labels),'valid':sum(r['group']==g for r in valid),'correct':sum(r['result']['labels'].get(k)==v for r in valid if r['group']==g for k,v in r['expected'].items())} for g in ['banking','routing','boolean','scoring']}}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'complete_core':summary['complete_core'],'quality':summary['quality']},indent=2))
if __name__=='__main__':main()
