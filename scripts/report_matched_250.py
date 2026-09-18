"""Report and verify the complete matched 250-case timing run."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/decision-engine/matched-latency-250-20260918'
DATA=json.loads((ROOT/'outputs/decision-engine/formal-benchmark-20260918/cases.json').read_text())
rows=[json.loads(l) for l in (OUT/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
s=json.loads((OUT/'summary.json').read_text())
protocol=json.loads((OUT/'protocol.json').read_text())
expected={c['id']:c for c in DATA['cases']}
assert len(rows)==750
for method in ('base','hf','ours'):
    rr=[r for r in rows if r['method']==method]
    assert len(rr)==250 and [r['case_id'] for r in rr]==protocol['case_order']
    assert len({r['case_id'] for r in rr})==250
    assert not any(r['result']['status']=='error' for r in rr)
    for r in rr:
        assert r['expected']==expected[r['case_id']]['expected']
        assert r['group']==expected[r['case_id']]['group']
        assert r['wall_ms']>0
    env=json.loads((OUT/f'environment-{method}.json').read_text())
    assert env['source_sha256']==json.loads((OUT/'environment-hf.json').read_text())['source_sha256']
    assert env['cases_sha256']==json.loads((OUT/'environment-hf.json').read_text())['cases_sha256']
    assert env['benchmark_sha256']==hashlib.sha256((ROOT/'scripts/benchmark_matched_250.py').read_bytes()).hexdigest()
lines=['# Matched latency on the same 250 questions','',
'All methods were freshly rerun on the same 250 inputs in the same seeded shuffled order, using the same local FP16 Qwen2.5-1.5B-Instruct checkpoint on the RTX 3050 Laptop 4 GB GPU. Only one GPU model process ran at a time. The engine implementation and model weights were not changed for this benchmark.','',
'## Main results','',
'| Method | Correct and schema-valid | Schema validity | Median request ms | Mean request ms | p95 request ms | Total for 250 s |',
'|---|---:|---:|---:|---:|---:|---:|']
for method,name in [('base','Base Qwen, unconstrained probability JSON'),('hf','[harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) (PyTorch/CUDA)'),('ours','Current optimized engine, graphs enabled')]:
    x=s[method]
    lines.append(f"| {name} | {x['correct']}/250 ({x['accuracy']:.1%}) | {x['schema_validity']:.1%} | {x['median_ms']:.1f} | {x['mean_ms']:.1f} | {x['p95_ms']:.1f} | {x['total_seconds']:.1f} |")
lines += ['', 'Accuracy counts a request as successful only if its required structured output is valid and its choice is correct. Base Qwen failures to generate the complete probability JSON count as failures; this is not a measurement of its general intelligence. Timing includes invalid and truncated attempts.','',
'## What the latency means','',
'Model loading is excluded. Each method received runtime initialization and three untimed requests from the separate robustness set. Warmup-specific optimization caches were then cleared once. All 250 measured inputs were sent once with caches allowed to persist normally through the stream. End-to-end synchronized wall time includes tokenization, inference, probability extraction and common schema validation. For our engine, new CUDA graph captures and evictions are included. This is model-resident latency for varied inputs, not exclusively pre-captured/repeated-shape latency.','',
'There is one timed attempt per input per method. Reported percentiles describe variation across the 250 tasks, not repeated-trial uncertainty. Methods ran in separate sequential blocks (HF, ours, base), so laptop thermal and system drift remain possible. Fresh-process startup was not benchmarked.','',
'## Accuracy and latency by task group','',
'| Group | Base correct / median ms | HF correct / median ms | Ours correct / median ms |','|---|---:|---:|---:|']
for g in ['banking','routing','boolean','scoring']:
    cols=[]
    for method in ['base','hf','ours']:
        x=s[method]['groups'][g];cols.append(f"{x['correct']}/{x['requests']} / {x['median_ms']:.1f}")
    lines.append('| '+g+' | '+' | '.join(cols)+' |')
lines += ['', '## Graph preparation within our measured stream','',json.dumps(s['ours']['graph_totals']), '']
for group,label in [('with_capture','Requests that captured a graph'),('without_capture','Requests without a graph capture')]:
    if group in s['ours']:
        x=s['ours'][group];lines.append(f"- {label}: {x['requests']}, median {x['median_ms']:.1f} ms.")
lines += ['', 'These subsets contain different questions and must not be used alone to claim a matched speedup. A single prepared routing example from an earlier run is not substituted for this 250-input result.','',
'## Scope and reproducibility','',
'The cases contain 100 BANKING77 messages from ten intents, 50 constructed routing examples, 50 explicit boolean judgments and 50 templated severity examples. These are the previously used benchmark cases, including cases used during prompt development; they are not a new unseen generalization test. All cases request one decision, so this suite does not measure scaling across many simultaneous fields.','',
'Artifacts: requests.jsonl contains all 750 timed responses, including raw generated baseline text and native engine metadata. summary.json contains aggregates; protocol.json records ordering and timing rules; environment-*.json record checkpoint, hardware, software and source hashes. Verification checked unique case coverage, identical ordering and expected answers, unchanged source hashes, and zero runtime errors.']
control_path=OUT/'no-graphs-summary.json'
if control_path.exists():
    ng=json.loads(control_path.read_text())
    controls=[json.loads(l) for l in (OUT/'no-graphs-requests.jsonl').read_text().splitlines()]
    assert len(controls)==250 and [x['case_id'] for x in controls]==protocol['case_order']
    assert all(x['result']['status']=='valid' for x in controls)
    assert all(x['result']['native']['metadata']['optimizations']['cuda_graphs'] is False for x in controls)
    ng_env=json.loads((OUT/'environment-ours-no-graphs.json').read_text())
    assert ng_env['source_sha256']==json.loads((OUT/'environment-ours.json').read_text())['source_sha256']
    assert ng_env['benchmark_sha256']==hashlib.sha256((ROOT/'scripts/benchmark_matched_250_no_graphs.py').read_bytes()).hexdigest()
    lines += ['', '## Follow-up control: CUDA graphs disabled', '',
    'After observing graph capture overhead in the primary run, we reran all 250 inputs with the existing CUDA graph option disabled. Input order, warmup, prompt, letter encoding, fused kernels, checkpoint, scoring and validation were preserved. This changes a runtime option, not model weights or engine source. It is a diagnostic follow-up selected after the main results.', '',
    '| Engine configuration | Correct and schema-valid | Schema validity | Median ms | Mean ms | p95 ms | Total for 250 s |',
    '|---|---:|---:|---:|---:|---:|---:|']
    for name,x in [('Graphs enabled',s['ours']),('Graphs disabled',ng)]:
        lines.append(f"| {name} | {x['correct']}/250 ({x['accuracy']:.1%}) | {x['schema_validity']:.1%} | {x['median_ms']:.1f} | {x['mean_ms']:.1f} | {x['p95_ms']:.1f} | {x['total_seconds']:.1f} |")
    lines += ['', f"Choices changed versus graphs enabled: {ng['choice_changes_from_graphs']}/250. Maximum absolute probability difference: {100*ng['max_probability_delta_from_graphs']:.3f} percentage points; {ng['probability_delta_over_1pp']}/250 exceeded one percentage point. These probabilities are uncalibrated token scores, not guaranteed correctness estimates.", '',
    'The control timings measure ordinary execution without graph capture or replay, with the other configured optimizations retained. This result is distinct from timing an already prepared graph on repeated shapes. No engine defaults were modified. Raw results and environment are in no-graphs-requests.jsonl, no-graphs-summary.json and environment-ours-no-graphs.json.']

(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps({'requests':len(rows),'same_case_order':True,'same_sources':True,'runtime_errors':0,'requests_sha256':hashlib.sha256((OUT/'requests.jsonl').read_bytes()).hexdigest()},indent=2)+'\n',encoding='utf-8')
print('\n'.join(lines[:17]))
