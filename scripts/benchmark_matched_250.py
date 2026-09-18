"""Matched 250-input, model-resident request timing; no engine changes."""
import argparse, gc, hashlib, json, random, statistics, time
from pathlib import Path
from run_formal_benchmark import ROOT, DATA, Runner, torch, environment, append, existing

OUT=ROOT/'outputs/decision-engine/matched-latency-250-20260918'
METHODS=('hf','ours','base')

def percentile(values,p):
    x=sorted(values); at=(len(x)-1)*p; lo=int(at); hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(at-lo)

def summarize():
    rows=existing(OUT/'requests.jsonl'); result={}
    for method in METHODS:
        rr=[r for r in rows if r['method']==method]
        if not rr: continue
        def metrics(items):
            ts=[r['wall_ms'] for r in items]
            valid=[r for r in items if r['result']['status']=='valid']
            correct=sum(all(r['result']['answers'][k]['choice']==v for k,v in r['expected'].items()) for r in valid)
            return dict(requests=len(items),valid=len(valid),correct=correct,accuracy=correct/len(items),schema_validity=len(valid)/len(items),median_ms=statistics.median(ts),mean_ms=statistics.mean(ts),p95_ms=percentile(ts,.95),total_seconds=sum(ts)/1000,errors=sum(r['result']['status']=='error' for r in items))
        result[method]=metrics(rr)
        result[method]['groups']={g:metrics([r for r in rr if r['group']==g]) for g in sorted({r['group'] for r in rr})}
        if method=='ours':
            for label,pred in [('with_capture',lambda r:r['graph_delta']['captures']>0),('without_capture',lambda r:r['graph_delta']['captures']==0)]:
                subset=[r for r in rr if pred(r)]
                if subset:result[method][label]=metrics(subset)
            result[method]['graph_totals']={k:sum(r['graph_delta'][k] for r in rr) for k in rr[0]['graph_delta']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cases=list(DATA['cases']);random.Random(20260918).shuffle(cases)
    assert len(cases)==250 and len({c['id'] for c in cases})==250
    protocol={'methods':list(METHODS),'model':'Qwen/Qwen2.5-1.5B-Instruct','cases':250,'case_order':[c['id'] for c in cases],'seed':20260918,'timing':'Synchronized whole request wall time, including tokenization, native inference, score extraction and schema validation. Model loading excluded. One timed attempt per case per method, same order for all methods.','warmup':'Engine runtime initialization then three untimed requests from the separate robustness set using the target method. Optimization caches cleared once afterward to exclude warmup-specific shapes. No per-case warmup, answer memoization, or repeated-input advantage. Graph captures/evictions during the measured stream count in latency.','methods_are_separate_processes':True,'output_contract':'Same common choice + all probabilities contract and validation as original formal benchmark. Invalid or truncated responses remain in timing denominator. Base is greedy unconstrained JSON generation with original output budget. HF pinned unmodified inference. Ours uses current delimited/letters production optimizations.','limitations':['One observation per case; percentiles describe this workload distribution, not repeated-trial jitter.','Method blocks run sequentially on one laptop, so thermal/system drift remains possible.','This measures a loaded model serving varied requests, not fresh-process startup or exclusively pre-captured shapes.']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    path=OUT/'requests.jsonl'
    if any(r['method']==a.method for r in existing(path)): raise RuntimeError('Method already has rows; refusing partial-cache resume or duplicate measurements')
    started=time.perf_counter();r=Runner();r.engine.prompt_format='delimited'
    assert r.engine.answer_encoding=='letters'
    r.engine.warmup()
    for case in DATA['robustness_cases'][:3]:
        warm=r.measured(a.method,case)
        if warm['result']['status']=='error':raise RuntimeError(str(warm['result']))
    r.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    env=environment();env.update({'method':a.method,'prompt_format':r.engine.prompt_format,'setup_seconds':time.perf_counter()-started,'benchmark_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/f'environment-{a.method}.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
    print('READY',a.method,'setup seconds',round(env['setup_seconds'],2),flush=True)
    for i,case in enumerate(cases):
        before=dict(r.engine._graphs.stats)
        row=r.measured(a.method,case)
        row.update({'group':case['group'],'expected':case['expected'],'sequence':i,'graph_delta':{k:v-before[k] for k,v in r.engine._graphs.stats.items()},'recorded_unix':time.time()})
        append(path,row)
        if (i+1)%10==0 or row['result']['status']=='error':
            summarize();print(a.method,i+1,'/250',case['id'],round(row['wall_ms'],1),'ms',row['result']['status'],flush=True)
        if row['result']['status']=='error':raise RuntimeError('Benchmark error; inspect saved row before continuing')
    print('COMPLETE',a.method,json.dumps(summarize()[a.method]),flush=True)

if __name__=='__main__':main()
