"""Check repository reproducibility and prevent premature public release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export_reviewer_interface import ROOT, build_payload, bundle_text


SOURCE_REQUIRED_FILES = (
    Path(".gitattributes"),
    Path(".github/workflows/ci.yml"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("REPOSITORY_RELEASE.md"),
    Path("reviewer_ui/README.md"),
    Path("reviewer_ui/dist/index.html"),
    Path("reviewer_ui/dist/styles.css"),
    Path("reviewer_ui/dist/app.js"),
)
FULL_ARTIFACT_FILES = (
    Path("paper/MANUSCRIPT_DRAFT.md"),
    Path("paper/MANUSCRIPT_DRAFT.docx"),
    Path("paper/RESULTS_MANIFEST.json"),
    Path("reviewer_ui/dist/review-data.js"),
)


def check(scope: str, root: Path = ROOT) -> dict:
    if scope not in {"source", "private", "public"}:
        raise ValueError("scope must be source, private or public")
    root = root.resolve()
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    required = SOURCE_REQUIRED_FILES + (() if scope == "source" else FULL_ARTIFACT_FILES)
    missing = [str(path) for path in required if not (root / path).is_file()]
    checks["required_repository_files_present"] = not missing
    if missing:
        blockers.append("missing required files: " + ", ".join(missing))

    if scope != "source" and not missing:
        payload = build_payload(root)
        expected = bundle_text(payload)
        actual = (root / "reviewer_ui/dist/review-data.js").read_text(encoding="utf-8")
        checks["reviewer_bundle_matches_frozen_inputs"] = actual == expected
        checks["reviewer_bundle_has_no_model_outputs"] = payload["model_outputs_included"] is False
        checks["reviewer_bundle_has_12_pairs_24_items"] = (
            len(payload["pairs"]) == 12 and len(payload["packet_rows"]) == 24
        )
        if not checks["reviewer_bundle_matches_frozen_inputs"]:
            blockers.append("reviewer bundle is stale; regenerate it before pushing")
        if not checks["reviewer_bundle_has_no_model_outputs"]:
            blockers.append("reviewer bundle unexpectedly includes model outputs")
        if not checks["reviewer_bundle_has_12_pairs_24_items"]:
            blockers.append("reviewer bundle item counts changed")

    manifest_path = root / "paper/RESULTS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    checks["working_manifest_declares_pending_human_review"] = "pending" in str(
        manifest.get("human_review", "")
    ).casefold() if manifest else False

    if scope == "public":
        public_requirements = {
            "manual_gold_and_receipt_present": (
                root / "data/holdout/version_gold_adjudicated.jsonl"
            ).is_file() and (
                root / "data/holdout/version_gold_adjudicated.manifest.json"
            ).is_file(),
            "holdout_execution_lock_present": (root / "paper/HOLDOUT_EXECUTION_LOCK.json").is_file(),
            "holdout_answer_scores_present": (root / "results/holdout_answer_scores.json").is_file(),
            "manifest_no_longer_marks_human_review_pending": not checks[
                "working_manifest_declares_pending_human_review"
            ],
        }
        checks.update(public_requirements)
        blockers.extend(
            f"public release gate not satisfied: {name}"
            for name, passed in public_requirements.items() if not passed
        )

    return {
        "status": "pass" if not blockers else "blocked",
        "scope": scope,
        "checks": checks,
        "blockers": blockers,
        "recommendation": (
            "Source-only repository is ready to push."
            if scope == "source" and not blockers
            else
            "Repository is ready for a private remote."
            if scope == "private" and not blockers
            else "Keep the remote private until every public-release gate passes."
            if scope == "public" and blockers
            else "Repository preflight passed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("source", "private", "public"), default="source")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = check(args.scope, args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
