"""Aggregate actual same-GPU comparison artifacts and disclose their limits."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/decision-engine/comparison-20260918'
d=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
extra=OUT/'long-baselines.json'
if extra.exists():
 followup=json.loads(extra.read_text(encoding='utf-8'));d['cases']['support_28']['methods'].update(followup['methods']);d['followup']=followup['reason']
lines=['# Current engine versus ordinary Qwen and the HF replica','',
'Actual local measurements on the same RTX 3050 Laptop 4 GB GPU, Linux/WSL, Qwen/Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, FP16 weights, PyTorch 2.6.0+cu124 and Transformers 5.5.4. No additional model training.','',
'## Current engine status','',
'The dedicated project contains a working decision-inference prototype: workflow JSON, shared context prefill, batched branches, selected answer-token scoring, cache optimizations, bounded CUDA graphs, and optional Triton RMSNorm/SwiGLU/RoPE fusion. Eighteen regression tests passed in the implementation task. Fusion is validated for dense Qwen2 with the pinned Linux/CUDA environment; graphs require repeated matching shapes for warm gains. The code is uncommitted. This is not yet a validated/calibrated production classifier or a universal hardware-autotuning runtime.','',
'The earlier 1.78x and 1.69x results compared our optimized engine with our own original engine. That original engine already had parallel branching and direct scoring. They were not speed ratios versus ordinary autoregressive Qwen.','',
'## Timing results','',
'Warm externally synchronized complete-request wall time in milliseconds. Three samples per method unless explicitly stated otherwise below. Model loading, graph/JIT setup and warmup are excluded. Samples are consecutive within each method, and method order rotates across workloads. Methods do not run concurrently.','',
'| Workload | Ordinary Qwen generating JSON | Independent per-question scoring | HF replica parallel CUDA | Our optimized engine |','|---|---:|---:|---:|---:|']
ratios={}
for name,c in d['cases'].items():
 cells=[]
 for mode in ('qwen_json','qwen_independent_scoring','hf_parallel','our_optimized'):
  m=c['methods'].get(mode)
  if not m:cells.append('not completed');continue
  if m['status']!='ok':cells.append(m['status']);continue
  flag=''
  if mode=='qwen_json':
   statuses={x['result']['status']for x in m['rows']}
   if statuses!={'valid'}:flag=' ('+', '.join(sorted(statuses))+')'
  if len(m['rows'])==1:flag+=' [one sample]'
  cells.append(f"{m['median_ms']:.1f}"+flag)
 lines.append('| '+name+' | '+' | '.join(cells)+' |')
 own=c['methods'].get('our_optimized',{})
 if own.get('status')=='ok':
  ratios[name]={mode:m['median_ms']/own['median_ms']for mode,m in c['methods'].items()if m['status']=='ok'and mode!='our_optimized'and (mode!='qwen_json'or all(x['result']['status']=='valid'for x in m['rows']))}
lines+=['','| Workload | Our speedup vs valid ordinary JSON | Our speedup vs HF CUDA |','|---|---:|---:|']
for n,rs in ratios.items():
 lines.append('| '+n+' | '+(f"{rs['qwen_json']:.2f}x"if 'qwen_json'in rs else'not established')+' | '+(f"{rs['hf_parallel']:.2f}x"if 'hf_parallel'in rs else'not established')+' |')
lines+=['',
'The small workloads favor our engine. On support_28, HF measured 1909–1989 ms while our engine measured 1701–2544 ms; the median difference is only about 3.5%, within these observed timing ranges. Treat that result as approximately tied, not an established performance advantage. Earlier stand-alone implementation timings are not substituted for this fresh comparison.','',
'## Output validity and quality','',
'| Workload / method | Valid generation runs | Labeled smoke score | Generated tokens |','|---|---:|---:|---:|']
for n,c in d['cases'].items():
 for mode,m in c['methods'].items():
  if m['status']!='ok':continue
  rows=m['rows'];q=rows[0].get('quality');quality=f"{q['correct']}/{q['fields']}"if q else'no verified labels'
  valid=f"{sum(x['result'].get('status')=='valid'for x in rows)}/{len(rows)}"if mode=='qwen_json'else'assembled structured output'
  toks=', '.join(str(x['result'].get('generated_tokens',0))for x in rows)if mode=='qwen_json'else'0'
  lines.append(f'| {n} / {mode} | {valid} | {quality} | {toks} |')
lines+=['',
'All methods got the one-field routing example correct. On the four-field refund example all methods got 3/4 correct, but they made different mistakes: our direct-scoring paths missed the explicit refund request; ordinary JSON and HF parallel chose severity 2 instead of the supplied expected severity 1. Equal scores on four fields do not prove equal general accuracy.','',
'For the 28-field support case there are no verified labels. Answers disagree between the HF and our engine, so this is a practical workload-speed comparison, not proof of interchangeable classifications.','',
'## Upstream candidate-token audit','',
'The downloaded HF PyTorch implementation compiles each enum choice to a candidate first token after stripping a shared character prefix. It computes collision flags, but run_parallel_generation_torch does not use those flags to perform continuation disambiguation. On the support workload the actual tokenizer mapping contains these collisions:','']
for field,x in d['cases']['support_28']['hf_candidate_collisions'].items():
 groups={}
 for option,token in zip(x['choices'],x['candidate_token_ids']):groups.setdefault(token,[]).append(option)
 for token,options in groups.items():
  if len(options)>1:lines.append(f'- {field}: {options} map to the same candidate token ID {token}.')
lines+=['',
'Those colliding choices receive identical logits in that implementation; argmax ties favor the earlier choice. Our A–Z answer IDs avoid these particular collisions. This is a correctness/coverage difference, not a reason to infer a general accuracy advantage from the latency table. The timing above runs the downloaded upstream functions unchanged.','',
'## Comparison controls and limitations','',
'- The same resident model weights, GPU and FP16 precision are supplied to all paths. The HF module receives the already-loaded model/tokenizer through its existing module cache; its inference/schema functions were not edited. Its default loader may otherwise select BF16 on this GPU.','- Our graph buffers are cleared between methods, so the HF and ordinary baselines do not run while graph memory is retained. Fusion adapters are inactive for the baseline/HF calls; the model-local forwarding wrappers remain installed after first use.','- Ordinary Qwen is standard greedy generation of labels-only JSON. It does not return per-option distributions. The prompt differs from direct scoring. It is the same instruct checkpoint, not a separate pretrained non-instruct Qwen model.','- The HF support case uses its pinned original schema/context. Refund cases translate our question definitions into its enum/boolean schema and retain option descriptions in the field descriptions. Different native prompt and scoring strategies remain part of each implementation.','- The initial comparison had a 600-second overall time limit and stopped during the long ordinary-generation repetitions before that method was checkpointed. The follow-up records one full 28-field generation sample after a 16-token generation warmup, plus three independent-scoring samples. Initial unsaved elapsed-only observations are not used for validity or ratios.','- All ordinary generation is capped at 768 new tokens. Invalid or truncated generations are marked and excluded from speedup-vs-valid-JSON claims. A failed/truncated fast response is not counted as equivalent useful output.','- These are small local timing samples with laptop scheduling/power variation. Cold capture and exact-shape reuse restrictions still apply. No statistical sub-percent accuracy claim is supported.','',
'## Source and artifacts','',
'[HF replica source at the exact tested revision](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/2af86848be75847ccb3553b0941cc51d6ef7e4e9). The upstream model card also publishes Apple M4 Max / 4-bit MLX results; those are a different backend, precision and device and are not directly compared with our CUDA timings.','',
'- comparison.json: original complete methods, full answers, timings, quality checks and candidate collision audit.','- long-baselines.json: checkpointed follow-up of the long generation and independent-scoring baselines.','- combined-comparison.json: merged data used by this report, preserving the separate raw artifacts.','- replica-source/manifest.json: exact upstream revision and SHA256 hashes; original public source files are preserved alongside it.','- scripts/compare_qwen_replica.py and scripts/complete_long_baseline.py: executable measurement harnesses.','- scripts/summarize_comparison.py: regenerate this report.','']
d['derived_speedups']=ratios
(OUT/'combined-comparison.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(OUT/'REPORT.md')