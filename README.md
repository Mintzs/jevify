# Jevify

Use an existing language model for fast classification, yes/no judgments, and rubric scoring. Define the allowed answers in `workflow.json`, send new context with each request, and receive structured JSON with scores for those answers.

Jevify reuses shared context, evaluates independent questions in batches, and constructs JSON directly from model scores. Optional GPU kernels reduce execution overhead. The default model is **Qwen2.5-1.5B-Instruct**; no fine-tuning is required.

## Install

Requires **Python 3.11+**. NVIDIA CUDA is recommended for speed; CPU execution is also supported.

```bash
git clone https://github.com/Mintzs/jevify.git
cd jevify
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **Linux / macOS:** `source .venv/bin/activate`

For NVIDIA GPUs, install a compatible [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/) first. Then install Jevify:

```bash
python -m pip install -e ".[inference]"
```

The first run downloads the model (roughly 3 GB). Later runs reuse the cached weights.

## Try it

From the repository folder:

```bash
jevify --interactive
```

Once the model is ready, enter a message and finish with `/run` on its own line:

```text
I was charged twice. Please refund the duplicate charge.
/run
```

Press Enter at each question prompt to use the saved instructions. Jevify prints JSON, then waits for another request. Type `/quit` to exit.

For a single request:

```bash
jevify --context "I was charged twice. Please refund the duplicate charge."
```

Use `--device cuda` or `--device cpu` to select a device. `python -m jevify` works as an alternative to the `jevify` command.

## Configure your workflow

Edit **[workflow.json](workflow.json)** to define the questions and allowed answers:

| Type | Purpose |
|---|---|
| `choice` | Pick from named options, such as a department or model route. |
| `noul` | Evaluate a yes/no question and return a probability. |
| `score` | Score the input against an ordered rubric. |

The included workflow demonstrates refund triage. Replace its questions and criteria for your application. Keep the rubric in the file and supply fresh context or conversation history with each request.

To use a different rubric, pass `--workflow path/to/workflow.json`. See [dynamic inputs and routing examples](docs/dynamic-inputs.md) for prompt overrides, request files, and conversation history.

## Use in Python

```python
from jevify import Workflow
from jevify.engine import DecisionEngine

engine = DecisionEngine.from_pretrained(device="auto")
workflow = Workflow.load("workflow.json")

result = workflow.run(
    engine,
    context="I was charged twice. Please refund the duplicate charge.",
)
print(result["answers"])
```

Keep the engine loaded between requests to avoid repeated model startup.

## Performance and limits

- Shared-context caching and batched question scoring are built in. Eligible Linux/CUDA configurations can use specialized shared-attention kernels.
- Additional Triton kernels are opt-in and require Linux/WSL, CUDA, and the tested Qwen2 setup. CUDA graphs are also opt-in; they can help repeated shapes but add capture overhead. See [optimization setup](docs/optimizations.md) and [branch execution](docs/branch-optimizations.md).
- Scores are **uncalibrated model probabilities**, not guarantees of correctness. Valid JSON can still contain a wrong decision.
- Qwen2.5-1.5B is the validated checkpoint. Other supported Qwen3/Llama backbones need their own verification. The engine currently uses one device and does not provide multi-GPU serving or quantization.

See the [technical guide](docs/decision-engine.md) for supported limits and the [answer-scoring guide](docs/literal-labels.md) for decoding options.

## Development

```bash
python -m unittest discover -s tests -p "test*.py" -v
```

GPU/Triton checks skip when their requirements are unavailable. Historical benchmark checks require local archived data and skip in a fresh checkout. Scripts under `scripts/` include experiments that depend on those local artifacts.

Downloaded weights, environments, raw benchmark outputs, and the video workspace are excluded from Git.
