"""Download pinned public benchmark inputs, not executable replica code.

The source declares Apache-2.0. Original bytes and hashes are preserved.
These scenarios have no verified labels and are latency workloads only.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPOSITORY = "harshatheg/Qwen-2.5-1B-RLCD"
REVISION = "2af8684"
NAMES = ("fintech_fraud", "code_security", "support_triage", "high_cardinality_255")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/decision-engine/replica-presets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(urlopen(f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}", timeout=30).read())
    revision = metadata["sha"]
    manifest = {"repository": REPOSITORY, "revision": revision, "declared_license": "apache-2.0",
                "verified_ground_truth": False, "files": [],
                "notes": ["Current presets may differ from those used in published timing tables.",
                          "No malformed source context is silently repaired.",
                          "Conversion changes the prompting/scoring implementation; it is not an exact replica benchmark."]}
    for name in NAMES:
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/presets/{name}.json"
        raw = urlopen(url, timeout=30).read()
        (args.output_dir / f"{name}.original.json").write_bytes(raw)
        source = json.loads(raw)
        questions = {}
        unsupported = []
        for key, field in source["schema"].items():
            if field["type"] == "boolean":
                questions[key] = {"type": "noul", "instructions": field["description"]}
            elif field["type"] == "enum" and len(field["choices"]) <= 26:
                questions[key] = {"type": "choice", "instructions": field["description"],
                                  "criteria": {choice: choice for choice in field["choices"]}}
            else:
                unsupported.append(key)
        item = {"name": name, "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                "field_count": len(source["schema"]), "unsupported_fields": unsupported}
        if not unsupported:
            request = {"context": source["context"], "questions": questions,
                       "source": {**item, "revision": revision, "declared_license": "apache-2.0"}}
            (args.output_dir / f"{name}.request.json").write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        manifest["files"].append(item)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
