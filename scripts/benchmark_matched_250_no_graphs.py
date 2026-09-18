"""One controlled follow-up: identical 250 inputs with optional graphs disabled."""
import gc,hashlib,json,time
from pathlib import Path
from run_formal_benchmark import ROOT,DATA,Runner,torch,environment,append
from benchmark_matched_250 import percentile
import statistics
OUT=ROOT/'outputs/decision-engine/matched-latency-250-20260918'

def main():
    path=OUT/'no-graphs-requests.jsonl'
    if path.exists():raise RuntimeError('Refusing to overwrite or resume a partial benchmark')
    protocol=json.loads((OUT/'protocol.json').read_text())
    by_id={c['id']:c for c in DATA['cases']};cases=[by_id[i] for i in protocol['case_order']]
    started=time.perf_counter();r=Runner();r.engine.prompt_format='delimited';r.engine.cuda_graphs=False
    r.engine.warmup()
    for c in DATA['robustness_cases'][:3]:
        warm=r.measured('ours',c)
        if warm['result']['status']=='error':raise RuntimeError(str(warm['result']))
    r.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    env=environment();env.update({'method':'ours_no_graphs','prompt_format':'delimited','cuda_graphs':False,'fused_kernels':list(r.engine.fused_kernels),'setup_seconds':time.perf_counter()-started,'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol':'Same input order, warmup and timing boundaries as primary run. Only CUDA graphs disabled. Secondary control selected after observing primary graph capture overhead.'})
    (OUT/'environment-ours-no-graphs.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
    print('READY ours_no_graphs',flush=True)
    rows=[]
    for i,c in enumerate(cases):
        row=r.measured('ours',c);row.update({'method':'ours_no_graphs','group':c['group'],'expected':c['expected'],'sequence':i,'recorded_unix':time.time()})
        append(path,row);rows.append(row)
        if (i+1)%25==0:print('ours_no_graphs',i+1,'/250',round(row['wall_ms'],1),'ms',row['result']['status'],flush=True)
        if row['result']['status']=='error':raise RuntimeError(str(row['result']))
    graph_rows={x['case_id']:x for l in (OUT/'requests.jsonl').read_text().splitlines() if (x:=json.loads(l))['method']=='ours'}
    def metrics(rr):
        ts=[x['wall_ms'] for x in rr];valid=[x for x in rr if x['result']['status']=='valid']
        ncorrect=sum(all(x['result']['answers'][k]['choice']==v for k,v in x['expected'].items()) for x in valid)
        return dict(requests=len(rr),valid=len(valid),correct=ncorrect,accuracy=ncorrect/len(rr),schema_validity=len(valid)/len(rr),median_ms=statistics.median(ts),mean_ms=statistics.mean(ts),p95_ms=percentile(ts,.95),total_seconds=sum(ts)/1000,errors=sum(x['result']['status']=='error' for x in rr))
    summary=metrics(rows);summary['groups']={g:metrics([x for x in rows if x['group']==g]) for g in sorted({x['group'] for x in rows})}
    summary['choice_changes_from_graphs']=sum(any(a['choice']!=graph_rows[x['case_id']]['result']['answers'][k]['choice'] for k,a in x['result']['answers'].items()) for x in rows)
    deltas=[max(abs(p-graph_rows[x['case_id']]['result']['answers'][k]['probabilities'][label]) for k,a in x['result']['answers'].items() for label,p in a['probabilities'].items()) for x in rows]
    summary['max_probability_delta_from_graphs']=max(deltas);summary['probability_delta_over_1pp']=sum(d>.01 for d in deltas)
    assert environment()['source_sha256']==json.loads((OUT/'environment-ours.json').read_text())['source_sha256']
    (OUT/'no-graphs-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
