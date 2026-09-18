import json,statistics,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/formal-benchmark-20260918'
s=json.loads((OUT/'summary.json').read_text());env=json.loads((OUT/'environment.json').read_text())
assert s['complete_core'],'Full core benchmark is incomplete'
def fmt(v,d=3):return 'n/a' if v is None else f'{v:.{d}f}'
lines=['# Qwen decision-engine benchmark','',
'All runs use the same Qwen2.5-1.5B-Instruct checkpoint, FP16 weights, RTX 3050 Laptop 4 GB GPU, and WSL. Our engine uses its implemented CUDA graphs and custom kernels. HF is the actual pinned upstream PyTorch implementation. Ordinary Qwen generates JSON autoregressively. These are results for one small checkpoint and one GPU, not claims of Jev-equivalent intelligence or universal acceleration.','',
'## Core decision quality: 250 cases','',
'Usable correctness requires both a valid output and the correct choice. Conditional choice accuracy excludes invalid outputs and must be read with its denominator.','',
'| Method | Valid schema | Correct and usable | Accuracy among valid | Brier, valid only | Log loss, valid only |','|---|---:|---:|---:|---:|---:|']
for m,v in s['quality'].items():lines.append(f'| {m} | {v["valid"]}/{v["requests"]} ({100*v["schema_valid_rate"]:.1f}%) | {v["usable_correct"]}/{v["requests"]} ({100*v["usable_accuracy"]:.1f}%) | {fmt(None if v["accuracy_on_valid"] is None else 100*v["accuracy_on_valid"],1)}% | {fmt(v["brier_valid_only"])} | {fmt(v["log_loss_valid_only"])} |')
lines+=['','Brier and log loss are better when lower. Their coverage differs whenever schemas fail; the baseline’s conditional probability score must not be interpreted as a result on all cases.','', '### Results by task','', '| Task | Ours correct/total | HF correct/total | Ordinary JSON correct/total |','|---|---:|---:|---:|']
for g in ['banking','routing','boolean','scoring']:
 vals=[s['groups'][m][g] for m in ['ours','hf','base']];lines.append('| '+g+' | '+' | '.join(f'{v["usable_correct"]}/{v["requests"]}' for v in vals)+' |')
lines+=['','### Labels-only ordinary Qwen','']
l=s['label_only']
if l:lines += [f'On the same {l["n"]} core requests, ordinary labels-only generation returned {l["valid"]} valid responses and {l["usable_correct"]} correct usable choices. Median attempted-request time: {l["median_ms"]:.1f} ms. This simpler contract has no probability output and is not the primary like-for-like comparison.','']
lines+=['### Probability comparison on identical valid cases','','| Method | Common-valid cases | Brier | Log loss |','|---|---:|---:|---:|']
for m,v in s['common_valid_quality'].items():lines.append(f'| {m} | {v["requests"]} | {fmt(v["brier_valid_only"])} | {fmt(v["log_loss_valid_only"])} |')
lines+=['','This intersection is selected by successful schema production, not a representative independent sample. Full calibration bins, five-bin ECE, and expected-level scoring error are in summary.json. No calibration guarantee is established.','',
'## Warm speed and memory','',
'Each cell records one first-shape call and five warm calls. Medians below include attempted calls; valid counts prevent failures from being mistaken for successful speedups. Five samples make p95 exploratory. Different context lengths and 1/4/16/28 questions are tested. The latency workload repeats three question types, not 28 separately labeled production tasks.','',
'| Workload | Ours ms (valid/n) | HF ms (valid/n) | Ordinary JSON ms (valid/n) |','|---|---:|---:|---:|']
for case,methods in s['speed'].items():
 vals=[]
 for m in ['ours','hf','base']:
  v=methods[m];vals.append(f'{fmt(v["median_ms"],1)} ({v["warm_valid"]}/{v["warm_n"]})')
 lines.append('| '+case+' | '+' | '.join(vals)+' |')
lines+=['','The full first-shape costs, empirical p95s, memory peaks, failure details, and valid fields/second are in summary.json and speed.jsonl. OOMs and truncations are retained. Do not claim a completed-task speedup against an invalid response. PyTorch allocated memory excludes driver, desktop, and reserved-pool overhead.','',
'## Fresh-process startup','',
'Three fresh worker processes per method; weights and compiler caches were already present and OS file caches were not flushed. Ready time includes initialization warmup for our engine. First-answer time includes imports, model loading, initialization, and the first real inference/capture. This is not model download time or first-ever compilation.','',
'| Method | Trials | Median ready seconds | Median entry-to-answer seconds | Status counts |','|---|---:|---:|---:|---|']
for m,v in s['cold'].items():lines.append(f'| {m} | {v["n"]} | {fmt(statistics.median(v["ready_seconds"]) if v["n"] else None,2)} | {fmt(statistics.median(v["entry_to_first_answer_seconds"]) if v["n"] else None,2)} | {v["status_counts"]} |')
lines+=['','External process wall time, including interpreter startup and cleanup, is also saved per trial.','',
'## Robustness checks','', '| Method | Valid | Correct and usable |','|---|---:|---:|']
for m,v in s['robustness'].items():lines.append(f'| {m} | {v["valid"]}/{v["requests"]} | {v["usable_correct"]}/{v["requests"]} |')
lines+=['','These 15 checks include 12 reordered/reworded examples and three labels sharing token prefixes; they are excluded from the main 250-case scores. Upstream candidate-token collisions are recorded explicitly.','',
'## Limitations and reproducibility','',
'- The banking portion is 100 test messages from ten selected intents. Other cases are constructed policy examples, with templated severity cases. This is a useful local evaluation, not a broad production certification.',
'- Schema-valid outputs do not imply correct classifications. Observed 100% validity is evidence for this suite, not proof that every request will succeed.',
'- All systems use their own native prompt format. Ordinary generation is one fixed prompt with no grammar enforcement or repair; stronger prompting or constrained-generation runtimes are separate baselines.',
'- HF receives the same resident FP16 weights as our engine; its default loader could choose BF16. The first-token scoring limitations of the pinned replica remain unchanged.',
'- Quality-request timings include fresh graph capture because caches are cleared to isolate memory between methods. Use the separate speed table for warm latency.',
'- Core cases and templates were frozen before predictions. The 20 pilot cases are included once in the 250, not added twice. No post-benchmark model or prompt tuning was performed.',
'- Benchmark validation tests passed for missing/extra fields, wrong labels/probabilities, duplicate JSON keys, and frozen-case integrity.',
'', 'Files: PROTOCOL.md; cases.json/cases.sha256; dataset-source.json; source-manifest.json; environment.json; quality.jsonl; labels.jsonl; cold.jsonl; speed.jsonl; summary.json.',
'', 'Public dataset: [BANKING77 source and license](https://github.com/PolyAI-LDN/task-specific-datasets).',
'', 'Re-run scripts/run_formal_benchmark.py with --stage pilot/full/labels/cold/speed in the configured environment. scripts/run_remaining_benchmarks.py coordinates sequential stages. scripts/summarize_formal_benchmark.py and scripts/report_formal_benchmark.py produce the analysis. Raw result files are append-only except adding external cold-process timings. Existing results should be archived to a new run directory before an independent rerun.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
with (OUT/'speed-summary.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=['workload','method','median_ms','p95_ms','warm_n','warm_valid','first_shape_ms','peak_allocated_mib','valid_fields_per_second']);writer.writeheader()
 for case,methods in s['speed'].items():
  for m,v in methods.items():writer.writerow({'workload':case,'method':m,**{k:v[k] for k in writer.fieldnames if k not in ['workload','method']}})
print(OUT/'REPORT.md')

raw=[json.loads(x) for x in (OUT/'quality.jsonl').read_text(encoding='utf-8').splitlines()]
data=json.loads((OUT/'cases.json').read_text());lookup={c['id']:c for c in data['cases']+data['robustness_cases']}
with (OUT/'decisions.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=['case_id','group','method','status','expected','choice','correct','gold_probability','selected_probability','wall_ms','error']);writer.writeheader()
 for row in raw:
  target=row['expected']['decision'];answer=row['result'].get('answers',{});answer=answer.get('decision',{}) if isinstance(answer,dict) else {}
  probs=answer.get('probabilities',{}) if isinstance(answer,dict) else {};choice=answer.get('choice') if isinstance(answer,dict) else None
  if not isinstance(probs,dict):probs={}
  writer.writerow({'case_id':row['case_id'],'group':row['group'],'method':row['method'],'status':row['result']['status'],'expected':target,'choice':choice,'correct':row['result']['status']=='valid' and choice==target,'gold_probability':probs.get(target),'selected_probability':probs.get(choice) if isinstance(choice,str) else None,'wall_ms':row['wall_ms'],'error':row['result'].get('error')})
wrong=['# Incorrect decisions with valid schemas','','These examples retain frozen gold labels. They are errors on this benchmark; public-dataset labels can still be ambiguous for short messages. No examples were removed after seeing predictions.','']
for method in ['ours','hf']:
 wrong+=['## '+method,'']
 errors=[r for r in raw if r['method']==method and r['result']['status']=='valid' and r['result']['answers']['decision']['choice']!=r['expected']['decision']]
 for r in errors[:15]:
  case=lookup[r['case_id']];answer=r['result']['answers']['decision'];wrong += [f"- **{r['case_id']}**: {case['context']}",f"  Expected `{r['expected']['decision']}`; selected `{answer['choice']}` with score {answer['probabilities'][answer['choice']]:.4f}.",'']
(OUT/'MISTAKES.md').write_text('\n'.join(wrong)+'\n',encoding='utf-8')
