"""Fresh, single-case traces for an honest side-by-side JSON replay."""
import gc,hashlib,json,time
from pathlib import Path
from run_formal_benchmark import ROOT,DATA,Runner,torch,environment
OUT=ROOT/'outputs/decision-engine/json-video-trace-20260918'
CASE_ID='routing_001'  # Fixed before measurement: first routing case, no latency-based selection.
class TokenTrace:
    def __init__(self,start):self.start=start;self.events=[];self.prompt=True;self.ended_ms=None
    def put(self,value):
        now=(time.perf_counter()-self.start)*1000
        if self.prompt:self.prompt=False;return
        values=value.tolist()
        assert all(isinstance(x,int) for x in values),'Single-sequence generation only'
        self.events.append({'at_ms':now,'token_ids':values})
    def end(self):self.ended_ms=(time.perf_counter()-self.start)*1000

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'trace.json'
    if target.exists():raise RuntimeError('Trace already exists; refusing to overwrite a recorded run')
    case=next(c for c in DATA['cases'] if c['id']==CASE_ID)
    protocol={'case_id':CASE_ID,'selection':'First routing case in the frozen 250-case suite, fixed before capture; illustrative single request, not the benchmark median.','methods':['ours','hf','base'],'timing':'Each method runs separately with one resident shared checkpoint. Device synchronized immediately before timing and after response. Includes prompt preparation, execution and validation; excludes model load and warmup. Replay aligns elapsed time zero; methods never competed on GPU.','base_stream':'Actual generate(streamer=...) token callbacks, timestamped when token IDs arrive on CPU. Prefix text decoded after completion. Streaming introduces instrumentation/CPU-transfer overhead; this is a streamed demonstration, not a replacement for the non-streaming 250-case benchmark.','decision_json':'Ours and HF show their common choice/probabilities API representation only after complete response. These JSON objects are assembled by code from model scores, not generated token by token. Native full responses retained.','ours_configuration':'delimited prompts, letters, CUDA graphs disabled, rmsnorm/swiglu/rope fusions enabled','display':'Real elapsed time at 1x. No invented intermediate JSON, re-timing or output repairs.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    runner=Runner();runner.engine.prompt_format='delimited';runner.engine.cuda_graphs=False;runner.engine.warmup()
    env=environment();env['trace_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env['fused_kernels']=list(runner.engine.fused_kernels);env['cuda_graphs']=False
    (OUT/'environment.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
    result={'case':case,'protocol':protocol,'environment':env,'traces':[]}
    for method in protocol['methods']:
        for warm_case in DATA['robustness_cases'][:3]:
            warm=runner.measured(method,warm_case)
            if warm['result']['status']=='error':raise RuntimeError(str(warm['result']))
        runner.engine.clear_optimization_cache();gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        original_generate=runner.model.generate
        start=time.perf_counter();stream=TokenTrace(start)
        if method=='base':
            def traced_generate(*args,**kwargs):
                return original_generate(*args,**kwargs,streamer=stream)
            runner.model.generate=traced_generate
        try:
            response=runner.call(method,case);torch.cuda.synchronize()
            wall_ms=(time.perf_counter()-start)*1000
        finally:runner.model.generate=original_generate
        tokens=[]
        for event in stream.events:
            tokens.extend(event['token_ids'])
            event['text']=runner.tokenizer.decode(tokens,skip_special_tokens=True)
        if method=='base':
            assert stream.events and stream.events[-1]['text']==response['raw_text']
            assert len(tokens)==response['generated_tokens']
            display=response['raw_text']
        else:display=json.dumps(response['answers'],indent=2,ensure_ascii=False)
        correct=all(response.get('answers',{}).get(k,{}).get('choice')==v for k,v in case['expected'].items())
        row={'method':method,'wall_ms':wall_ms,'token_events':stream.events,'stream_ended_ms':stream.ended_ms,'display_text':display,'display_format':'unmodified generated text' if method=='base' else 'common response schema (full precision, metadata omitted)','schema_valid':response['status']=='valid','correct_choice':correct,'result':response}
        result['traces'].append(row)
        (OUT/f'{method}.json').write_text(json.dumps(row,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        print('TRACE',method,round(wall_ms,2),'ms','schema',row['schema_valid'],'choice',correct,'tokens',len(tokens),flush=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print('COMPLETE',str(target),flush=True)
if __name__=='__main__':main()
