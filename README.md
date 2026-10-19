# Version-Aware Label RAG

Source code for reproducible, evidence-complete and version-aware question answering over FDA drug labels.

This repository contains the retrieval, structured SPL parsing, evaluation, provenance, review-validation and reporting code. Research data, holdout sources, reviewer exports, results and manuscript files are intentionally excluded from Git.

## Research boundary

The implementation separates:

- retrieval completeness for every atomic claim;
- citation location and textual claim support;
- answer completeness and refusal behavior;
- exact SPL version identity;
- development experiments from a sealed temporal holdout.

No clinical reliability claim should be inferred from the source code or synthetic tests.

## Repository layout

```text
src/labelqa/       Python research pipeline
tests/             Synthetic and unit tests
reviewer_ui/       Offline reviewer application source
.github/workflows  Cross-platform CI
```

## Requirements

- Python 3.10 or newer
- Node.js only for optional JavaScript syntax checking
- No required Python runtime dependencies outside the standard library

## Run the test suite

PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python -m labelqa.repository_preflight --scope source
```

The source preflight inspects the Git index and fails if it finds research data,
results, manuscripts, reviewer payloads, PDF files or DOCX files in the tracked tree.

Linux or macOS:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m labelqa.repository_preflight --scope source
```

## Run the synthetic pipeline demo

The demo uses an embedded fictional SPL document. It parses narrative and a table row, runs six retrieval methods, builds a provenance-bound prompt and mechanically validates a constructed claim/citation response. It makes no model call and contains no research or holdout data.

```powershell
$env:PYTHONPATH='src'
python -m labelqa.synthetic_demo --output results/synthetic-demo-report.json
```

A successful run returns `status: synthetic_smoke_passed`.

## Reviewer interface

The static application lives in `reviewer_ui/dist/`. Its generated `review-data.js` is excluded because it contains the private source packet. With authorized local data present, generate it using:

```powershell
$env:PYTHONPATH='src'
python -m labelqa.export_reviewer_interface
```

Then open `reviewer_ui/dist/index.html`.

The holdout has no human review. Its reference is an AI silver standard compiled by a fixed rule from two disclosed AI annotations (`labelqa.compile_silver_standard`), and holdout answers are scored mechanically (`labelqa.summarize_holdout_mechanical`); no clinical claim is made from it. AI reviewer IDs (for example `llm_…` or `codex_…`) can never receive a human receipt or sign a resolution. `python -m labelqa.rehearse_holdout_pipeline --design ai_silver --root <new-dir>` rehearses this sequence on development labels; the human-review designs remain available as `llm_expert` and `two_reader`.

The method-blind answer review has its own offline interface in `answer_review_ui/dist/`; its generated `answer-review-data.js` (model outputs on the holdout) is excluded from Git.

## Data and artifacts

The pipeline expects project-local `data/`, `results/` and `paper/` directories for full experiments. They are excluded from the source repository. Frozen inputs and outputs should be transferred through a controlled research archive with their SHA-256 manifests.

## Repository policy

See [REPOSITORY_RELEASE.md](REPOSITORY_RELEASE.md) for release checks and repository settings. A project license must be selected by the owner before a public release.
