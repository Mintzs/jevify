# Literal-label scoring

The optional `answer_encoding="labels"` / `--answer-encoding labels` mode scores the
actual option names in the workflow. For example, `billing`, `shipping`, and
`returns` replace A/B/C. No workflow edits are required. Descriptions still tell
the model what each option means. Noul uses false/true; Score uses its actual
zero-based level names ("0", "1", etc.). The external JSON schema is unchanged.

The neutral-context correction has been removed, including its CLI flag and
cache. The model's weights are unchanged. Probabilities are only normalized
among the allowed alternatives; no prior division or confidence adjustment is used.

## One-token labels

If every candidate is a single token, one decision position provides all scores.
The engine selects just the required LM-head rows, projects in FP32, and applies
softmax. Existing question batching, cache reuse, eligible shared-attention kernels,
and optional CUDA graphs remain available. Boolean scoring is unchanged.

## Longer labels

For a multi-token label, the engine evaluates the complete sequence likelihood:
the probability of each token given the prompt and the preceding label tokens.
It sums log probabilities and normalizes across the candidate sequences. It does
not normalize by length. Consequently, longer labels can receive lower scores;
actual labels do not guarantee better task accuracy or eliminate prompt bias.

The prompt is computed once per such question. Known candidate continuations are
teacher-forced in batches, reusing that prompt's KV cache. This is bounded scoring,
not free-text generation. Full-vocabulary normalization is necessary here; the
head uses the model's native dtype, with FP32 log-softmax and accumulation.
Projection is chunked to limit temporary GPU memory. This can be slower than
one-token selected-head scoring. Multi-token suffixes currently use ordinary
attention and eager execution, with the existing optional pointwise fusions.
They do not use the specialized shared-prefix attention or CUDA graphs. Prompt
prefill may use graphs; all-single-token question groups retain the fast paths.

Shared first tokens are supported. If one complete label's token sequence is a
prefix of another, EOS is appended to every candidate of that question so the
alternatives represent distinct completed answers. The default label limit is
64 tokens including required EOS (`--max-label-tokens`). Duplicate token sequences,
special tokens in labels, and overlength requests fail explicitly; the engine
never silently substitutes letters or truncates labels.

## Reproduction and metadata

Letters remain the default after the latency/accuracy comparison. Use
`--answer-encoding labels` to request literal-label scoring explicitly. `--prompt-format json` selects a JSON
option catalog; it does not change the chosen answer encoding.

Metadata reports each field's `answer_encoding` and `answer_token_lengths`.
Multi-token requests additionally report `label_scoring` and
`full_vocabulary_projection_positions`. Their probability status is
`uncalibrated_candidate_sequence_probabilities`; one-token requests use
`uncalibrated_answer_token_probabilities`. Neither is empirical calibration.

`python scripts/benchmark_literal_labels.py` runs the frozen paired CUDA comparison
in the prepared WSL environment. Results, real prompts, token IDs, source/data
hashes, and timing conditions are saved under
`outputs/decision-engine/natural-labels-20260918/`. The historical HF scores come
from the pinned original benchmark, not a modified HF implementation.

The literal-label benchmark used `--prompt-format readable`. The current production
default is `--prompt-format delimited`; specify readable explicitly to reproduce
that earlier comparison. See [current prompt format](prompt-format.md).
