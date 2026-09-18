# Automatic branch optimizations

The engine accepts the same workflow.json and returns the same typed answers.
These optimizations change execution, not model weights or task definitions.

## Default behavior

- **Length grouping:** when there are more questions than fit in one branch batch,
  the engine groups similar token lengths if doing so removes at least 10% of
  padded branch tokens. Results are restored to the original question order.
- **Shared physical prefix:** eligible batches read one prefix KV cache directly
  and keep only their own suffix keys/values. The Triton kernel applies exact
  causal attention with grouped query heads and per-branch lengths. It does not
  approximate attention, shorten context, or concatenate repeated prefix copies.
- **Short branches:** the shared kernel uses a smaller query tile only at 16
  suffix tokens or fewer. Longer branches use its general tile.
- **Bounded graph streams:** CUDA graph captures reuse one stream per model.
  This avoids accumulating cuBLAS workspaces as input shapes change. Existing
  graph entry/memory limits still apply. Graphs remain opt-in and exact-shape;
  this change does not remove capture startup cost.

Automatic shared attention currently requires the measured configuration:
Linux/CUDA, an SM 8.6 GPU, FP16 dense Qwen2, Transformers 5.5.4, SDPA, and Triton.
It requires at least two branches, at least 128 shared prefix tokens, no sliding
attention, head sizes 32/64/128, prefix length at most 4096, and suffix width at
most 512. Outside that envelope, the ordinary backend runs automatically. A
supported workflow does not need different field names, rubrics, or custom code.

`shared_attention="on"` permits other supported CUDA SM 8+ FP16/BF16 configurations
for explicit evaluation; the same structural limits still apply. It does not
force unsupported execution. Compilation/resource failures disable shared
attention for that engine and fall back; device execution errors are surfaced.
Broader model/GPU performance is not claimed from the RTX 3050 measurements.

Runtime sequence lengths and strides are passed into the attention kernel rather
than compiling a separate kernel for every context length. JIT compilation still
occurs on first use of a kernel configuration. No remote model code is loaded.

## Adjacent-operation fusion

`residual_norm` combines the post-attention residual addition and RMS normalization
in one Triton launch, preserving the reference intermediate low-precision rounding.
It is implemented and tested, but remains opt-in: isolated kernel measurements
improved while full-request results were mixed. It is not in the automatic profile.

The existing `rmsnorm`, `swiglu`, and `rope` options remain available. Shared
attention can run without those optional fusions. CUDA graphs are separately
controlled by `--cuda-graphs`; workload shape reuse determines their value.

## Controls

Default API parameters are `shared_attention="auto"`, `specialize_short=True`,
and `length_aware=True`. For a reference comparison use `shared_attention="off"`,
`specialize_short=False`, and `length_aware=False`.

The CLI provides `--shared-attention auto|off|on`, `--no-short-attention`, and
`--no-length-grouping`. Optional fusion can be selected with
`--fused-kernels rmsnorm swiglu rope residual_norm`. Ordinary workflow users do
not need to adjust these switches for each request.

Metadata reports shared/short branch counts, automatic fallback reasons, whether
questions were reordered, and padded token totals before/after grouping. The
cache_storage field reports shared, repeated, or mixed execution. Scores remain
uncalibrated answer-token probabilities; schema compliance is independent of
semantic classification accuracy.

## Evidence

The local report is `outputs/decision-engine/branch-optimization-20260918/REPORT.md`.
It includes individual-stage and alternating end-to-end timings, probability
changes, memory use, and rejected/default-off candidates. `profile-before.json`
contains the CUPTI kernel profile. `tests.json` records the final regression run.

The early `candidates-before-stream-fix.json` run is retained only as diagnostic
evidence: graph-stream workspace growth confounded its timings. It was stopped
before completion and is not used for accepted speedup claims. GPU microbenchmarks
also exhibited some large timing outliers; their results do not substitute for
full-request confirmation.
