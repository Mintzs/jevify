"""Sequential GPU stages; no overlapping resident models."""
import subprocess,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/formal-benchmark-20260918'
worker=ROOT/'scripts/run_formal_benchmark.py'
def run(args):
    print('START STAGE',args,flush=True);started=time.perf_counter()
    subprocess.run([sys.executable,'-u',str(worker),*args],check=True)
    return time.perf_counter()-started
run(['--stage','full'])
run(['--stage','labels'])
for trial in range(3):
    methods=['ours','hf','base'];methods=methods[trial:]+methods[:trial]
    for method in methods:
        path=OUT/'cold.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if any(x['method']==method and x['trial']==trial for x in rows):continue
        elapsed=run(['--stage','cold','--method',method,'--trial',str(trial)])
        rows=[json.loads(x) for x in path.read_text().splitlines()];rows[-1]['external_process_wall_seconds']=elapsed
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
run(['--stage','speed'])
subprocess.run([sys.executable,str(ROOT/'scripts/summarize_formal_benchmark.py')],check=True)
print('ALL BENCHMARK STAGES COMPLETE',flush=True)
