"""Check repository reproducibility and prevent premature public release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

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
SOURCE_FORBIDDEN_PREFIXES = ("data/", "paper/", "results/", "private_review/")
SOURCE_FORBIDDEN_PATHS = {"reviewer_ui/dist/review-data.js"}
SOURCE_FORBIDDEN_SUFFIXES = (".pdf", ".docx")


def prohibited_source_paths(paths: list[str]) -> list[str]:
    """Return tracked paths that violate the source-only repository boundary."""
    prohibited = []
    for raw_path in paths:
        path = raw_path.replace("\\", "/").lstrip("./")
        folded = path.casefold()
        if (folded.startswith(SOURCE_FORBIDDEN_PREFIXES)
                or folded in SOURCE_FORBIDDEN_PATHS
                or folded.endswith(SOURCE_FORBIDDEN_SUFFIXES)):
            prohibited.append(path)
    return sorted(set(prohibited))


def source_tree_paths(root: Path) -> tuple[list[str], str]:
    """Read the Git index when available, otherwise inspect an unpacked source tree."""
    try:
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
        )
    except OSError:
        top_level = None
        result = None
    if (top_level is not None and top_level.returncode == 0
            and Path(top_level.stdout.strip()).resolve() == root
            and result is not None and result.returncode == 0):
        return [part.decode("utf-8") for part in result.stdout.split(b"\0") if part], "git_index"
    return [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()], "filesystem"


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

    source_tree_inspection = None
    if scope == "source":
        tree_paths, inspection = source_tree_paths(root)
        source_tree_inspection = inspection
        prohibited = prohibited_source_paths(tree_paths)
        checks["source_tree_excludes_private_artifacts"] = not prohibited
        if prohibited:
            blockers.append("source-only tree contains prohibited artifacts: " + ", ".join(prohibited))

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

    if scope != "source":
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

    report = {
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
    if source_tree_inspection is not None:
        report["source_tree_inspection"] = source_tree_inspection
    return report


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
