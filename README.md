# Jevify

An experimental CUDA/PyTorch engine for parallel classification, yes/no judgments,
and rubric scoring with a shared context. The default model is
`Qwen/Qwen2.5-1.5B-Instruct`. The Python package is `ora_decision_engine`; the CLI is `ora-decision`.
This repository is independent of the Distillation project.

## Configure once, supply fresh inputs each time

Edit **workflow.json** to define question types, answer options, scoring criteria,
and optional default instructions. Supply context, conversation history, and
question overrides separately on every request. No engine code changes are needed.
The supplied default is a four-question refund rubric, not a validated policy.

## Run locally

From this repository's root, using the existing Windows CUDA environment:

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.cache/decision-engine/huggingface'
.\.cache\decision-engine\venv\Scripts\python.exe -m ora_decision_engine --interactive --device cuda --local-files-only
```

This loads and warms up the model once. Paste context, enter `/run` on its own line, then
supply question prompts or press Enter for the saved defaults. `/quit` exits.
For application integration, load `Workflow` and `DecisionEngine` once and call
`workflow.run(engine, prompt=..., context=..., messages=...)` with new inputs.
See [dynamic requests, conversation history, and routing examples](docs/dynamic-inputs.md).

One-shot requests accept `--context`, `--context-file`, or `--request` alongside
`--workflow` (default `workflow.json`). `--prompt` overrides a single question;
request JSON uses `prompts` for multiple named questions. The original combined
request format remains available through `--input examples/decision_engine/refund.json`.

For a fresh checkout, follow [installation and API documentation](docs/decision-engine.md).
Install the CUDA PyTorch build appropriate to your device, then `pip install -e '.[inference]'`.
The installed `ora-decision` command is equivalent to `python -m ora_decision_engine`.

## Tests and evidence

```powershell
.cache/decision-engine/venv/Scripts/python.exe -m unittest discover -s tests -p test_decision_engine.py -v
```

Historical benchmark results and captured raw model tensors are preserved locally
under `outputs/decision-engine/`. Model weights, virtual environments, and downloaded
wheels live under `.cache/decision-engine/`. Both directories are ignored by Git. The local video workspace under `videos/` is also excluded.
Historical benchmark integration tests require the local archived data and upstream
source snapshot; they skip when those artifacts are absent from a fresh checkout.
Source code, examples, the default workflow, tests, and documentation are trackable.

The default scores A/B/C option IDs for Choice and Score; Noul uses
`false`/`true`. Single-token answers retain selected-head scoring. With explicit label encoding, multi-token
labels are evaluated in full with a cached prompt and batched known continuations.
These are uncalibrated model likelihoods, not measured correctness probabilities.
Workflow files and answer structures are unchanged. See [literal-label scoring](docs/literal-labels.md).

Earlier comparisons remain in `outputs/decision-engine/`. The neutral-context
probability adjustment has been removed; its archived reports are historical.
Older results remain historical evidence. See [implementation and limitations](docs/decision-engine.md).

## Optional inference optimizations

The engine now includes bounded CUDA graph replay, reduced prefix-cache copying,
reused answer-head weights, and optional Triton RMSNorm, SwiGLU and RoPE kernels.
See [optimization setup and behavior](docs/optimizations.md).

## Automatic branch execution

Eligible batches now share one physical context cache, use a short-branch kernel
where appropriate, and group similar question lengths to reduce padding. Other
inputs automatically use ordinary attention. Workflow JSON needs no changes.
CUDA graph captures also reuse a stream to avoid workspace growth across shapes.
See [automatic dispatch, fallbacks, and measurements](docs/branch-optimizations.md).

## Routing prompt correction

Choice/Score options use JSON-escaped actual labels and descriptions. Interactive
sessions initialize the model before accepting requests; use `--no-warmup` to
skip that step. See [input and timing details](docs/dynamic-inputs.md).

## Literal-label benchmark

The paired comparison with uncorrected letter scoring is saved under
`outputs/decision-engine/natural-labels-20260918/`. The default is
`--answer-encoding letters`; `--answer-encoding labels` enables the literal-label
comparison path. No neutral-context recalculation is available.

## Clearer input and rubric separation

The default `--prompt-format delimited` wraps the input in `<input_text>` markers
and identifies the question/options as evaluation instructions. Users still supply
ordinary context and prompts; the engine adds these boundaries automatically.
Choice and Score keep one-token A/B/C IDs, and Noul keeps false/true. No neutral
probability correction or model training is used.

The controlled prompt comparison is recorded in
`outputs/decision-engine/prompt-bias-20260918/`. See [prompt behavior and validation](docs/prompt-format.md).
Use `--prompt-format readable --answer-encoding letters` for the previous template.
