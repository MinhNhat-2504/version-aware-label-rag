# Repository release guide

## Recommended repository name

`version-aware-label-rag`

Short description:

> Reproducible evidence-complete and version-aware QA over FDA drug labels, with claim-level citations and a sealed temporal holdout.

Alternative names, in preference order:

1. `evidence-complete-label-qa`
2. `spl-version-qa`
3. `fda-label-evidence-rag`

## Source-only repository

The GitHub commit contains source code, tests, CI, the static reviewer application and usage documentation. It excludes `paper/`, `data/`, `results/`, the original PDF and generated `review-data.js`. Keep research data and manuscripts in the local workspace.

Run this before every push:

```powershell
$env:PYTHONPATH='src'
python -m labelqa.repository_preflight --scope source
python -m unittest discover -s tests -v
```

The full-artifact public gate is deliberately expected to fail today:

```powershell
$env:PYTHONPATH='src'
python -m labelqa.repository_preflight --scope public
```

It remains blocked until manually adjudicated gold and its receipt, the frozen holdout execution lock, final holdout answer scores, and a results manifest without pending human review all exist.

## Suggested GitHub settings

- Visibility: Private.
- Default branch: `main`.
- Require the `CI` check before merging into `main`.
- Disable force pushes and branch deletion on `main`.
- Keep reviewer exports, datasets and manuscripts outside the repository.
- Choose the project license before making the repository public. Licensing is a project-owner decision and should cover only original code and documentation; the vendored FDARxBench snapshot retains its upstream license.

## First push

After creating an empty private repository on GitHub:

```powershell
git init
git branch -M main
git add .
git commit -m "Initialize version-aware FDA label QA research repository"
git remote add origin https://github.com/OWNER/version-aware-label-rag.git
git push -u origin main
```

Inspect `git status` and the staged file list before committing. The local research data, paper and results are ignored by the source-only repository.
