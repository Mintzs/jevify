"""Repeated warm captures for the minimal comparison; preserve every trial."""
import gc,hashlib,json,time,statistics
from pathlib import Path
from run_formal_benchmark import ROOT,DATA,Runner,torch,environment
from capture_json_video_trace import TokenTrace
OUT=ROOT/'outputs/decision-engine/json-video-trace-minimal-v2'
CASE_ID='routing_031'
def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'trace.json').exists():raise RuntimeError('Preserve previous capture')
    case=next(c for c in DATA['cases'] if c['id']==CASE_ID)
    protocol={'case_id':CASE_ID,'selection':'First routing case in frozen suite order with our correct valid answer; chosen from existing results before new timing. Successful demonstration, not an accuracy estimate. No selection for fastest latency.','repeats':5,'warmups_per_method':3,'trial_selection':'Median wall-time trial independently for each method; all trials retained.','timing':'Sequential GPU runs aligned at elapsed zero. Same loaded Qwen checkpoint, CUDA synchronized before and after each request. Includes prepare/inference/validation, excludes model load. Three same-case warmups before five timed requests. No cache clearing between those requests.','base_stream':'Actual generate streamer CPU token arrivals; added callback/transfer overhead is included. Other engines return assembled JSON, no token stream.','configuration':'delimited prompt, letters, graphs off, rmsnorm/swiglu/rope fusions on','playback_rate':1}
    save(OUT/'protocol.json',protocol)
    runner=Runner();runner.engine.prompt_format='delimited';runner.engine.cuda_graphs=False;runner.engine.warmup()
    env=environment();env['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();env['cuda_graphs']=False;env['fused_kernels']=list(runner.engine.fused_kernels)
    save(OUT/'environment.json',env)
    result={'case':case,'protocol':protocol,'environment':env,'traces':[],'all_trials':{}}
    for method in ['ours','hf','base']:
        runner.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache()
        for i in range(3):runner.measured(method,case)
        trials=[]
        for repeat in range(5):
            torch.cuda.synchronize();original=runner.model.generate;start=time.perf_counter();stream=TokenTrace(start)
            if method=='base':
                def traced(*args,**kwargs):return original(*args,**kwargs,streamer=stream)
                runner.model.generate=traced
            try:
                response=runner.call(method,case);torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
            finally:runner.model.generate=original
            tokens=[]
            for e in stream.events:
                tokens.extend(e['token_ids']);e['text']=runner.tokenizer.decode(tokens,skip_special_tokens=True)
            if method=='base':assert stream.events[-1]['text']==response['raw_text'] and len(tokens)==response['generated_tokens']
            correct=all(response.get('answers',{}).get(k,{}).get('choice')==v for k,v in case['expected'].items())
            row={'method':method,'repeat':repeat,'wall_ms':wall,'token_events':stream.events,'stream_ended_ms':stream.ended_ms,'display_text':response['raw_text'] if method=='base' else json.dumps(response['answers'],indent=2,ensure_ascii=False),'schema_valid':response['status']=='valid','correct_choice':correct,'result':response}
            trials.append(row);save(OUT/f'{method}-{repeat}.json',row)
            print(method,repeat,round(wall,2),'ms','schema',row['schema_valid'],'correct',correct,flush=True)
        selected=sorted(trials,key=lambda r:r['wall_ms'])[2];result['traces'].append(selected);result['all_trials'][method]=trials
        save(OUT/f'{method}.json',selected)
    # Compare the old video's exact case with the same current engine when warmed.
    old=next(c for c in DATA['cases'] if c['id']=='routing_001')
    for _ in range(3):runner.measured('ours',old)
    diag=[runner.measured('ours',old) for _ in range(5)];save(OUT/'old-case-warm-diagnostic.json',diag)
    save(OUT/'trace.json',result)
    print('SELECTED',[(r['method'],round(r['wall_ms'],2),r['correct_choice']) for r in result['traces']],flush=True)
    print('OLD CASE WARM MEDIAN',statistics.median(r['wall_ms'] for r in diag),flush=True)
if __name__=='__main__':main()
