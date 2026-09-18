# Prompt boundaries with one-token answers

The default is `prompt_format="delimited"` and `answer_encoding="letters"`.
The engine supplies this structure automatically:

```text
<input_text>
"The user's actual context, JSON-escaped to preserve its contents"
</input_text>

Evaluation instructions (not input text):
The workflow question or dynamic prompt
Options:
A. "label": "description"
B. "label": "description"
Return only the letter of the best option.
```

Noul uses the same separation and still asks for true/false. Score still presents
its numeric levels and descriptions mapped to one-token letters. Workflow files,
dynamic context/prompt/history APIs, answer schema, and model weights are unchanged.
There are no workflow-specific rules in the engine. The input markers are prompt
formatting, not an XML parser. Input strings retain their JSON escaping.

The model still sees every label and description. Only the answer is a compact
letter ID. The engine reads the corresponding token scores and normalizes within
the allowed set as before; no neutral-context division or score correction occurs.
Different prompt wording intentionally changes the model's judgments. These
probabilities remain uncalibrated.

The common context prefix still supports parallel question branches, KV reuse,
eligible shared-attention kernels, optional pointwise fusions, and CUDA graphs.
No extra model pass or generated explanation is added. Extra prompt tokens can
slightly change latency and the input budget. The full prompt must fit the configured
limit; the engine never truncates silently.

## Validation

Eight prompt/mapping alternatives were compared with the original on a frozen
80-case development subset. The selected delimiter format was then checked against
the entire original 250, the additional 80, 15 robustness cases, and 58 new cases
frozen before selection. The 170 original cases outside development improved too.
No expected labels were modified. Exact results, score vectors, timing samples,
data/source hashes and remaining failures are in
`outputs/decision-engine/prompt-bias-20260918/REPORT.md`.

The old readable template and JSON catalog remain explicitly available with
`--prompt-format readable` and `--prompt-format json`. Reproducing the literal-label
comparison requires `--prompt-format readable --answer-encoding labels`; that
comparison predates the new delimiter format. Literal labels remain optional
because multi-token labels add scoring work and did not improve the earlier
benchmark overall.
