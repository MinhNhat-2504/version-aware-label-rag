"""Freeze a small, source-grounded assistant-drafted temporal QA pilot.

These records are a review queue, not a validated evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_jsonl


EXPECTED_SET_IDS = (
    "03b39d40-90fe-11df-9de6-0002a5d5c51b",
    "092da413-ec60-4577-ac57-51e3e42d55c9",
    "0b0f0209-edbd-46f3-9bed-762cbea0d737",
    "164500d7-51cf-438a-8bee-7a7dce76ad7a",
    "16fb8bc7-1db1-477e-a201-b3b678b8a84c",
    "1aa75a3a-18c9-49e1-91a6-293d0b7da756",
)


# (pool row, task, question, reference answer, atomic claim / accepted source indices)
ITEMS = [
    (0, "factual", "What is the recommended ISTODAX dose and schedule?",
     "14 mg/m² intravenously over four hours on days 1, 8, and 15 of a 28-day cycle.",
     [("Dose, route, infusion duration and schedule", [4])]),
    (0, "multihop", "Which adult patients is ISTODAX indicated for, and what ECG-related precautions does its label describe?",
     "It is indicated for adults with cutaneous T-cell lymphoma after at least one systemic therapy. For patients with long QT syndrome, significant cardiovascular disease, or QT-prolonging drugs, consider ECG monitoring at baseline and during treatment; confirm normal potassium and magnesium before administration.",
     [("Indicated adult population and prior therapy", [2]), ("ECG monitoring and electrolyte precautions", [13])]),
    (1, "factual", "What maximum bupropion SR dose does the label give for moderate to severe hepatic impairment?",
     "100 mg per day or 150 mg every other day.",
     [("Maximum dose for moderate to severe hepatic impairment", [5, 37])]),
    (1, "multihop", "What does the bupropion SR label say about seizure disorder and dose adjustment for renal impairment?",
     "It is contraindicated in patients with a seizure disorder. For renal impairment (GFR below 90 mL/min), consider reducing the dose and/or dosing frequency.",
     [("Seizure disorder contraindication", [10, 14, 53]), ("Renal impairment dose consideration", [6, 36])]),
    (2, "factual", "Is lamotrigine recommended for an acute manic or mixed episode in this label?",
     "No. The label indicates maintenance treatment of bipolar I disorder and does not recommend treatment of acute manic or mixed episodes.",
     [("Maintenance indication and acute episode limitation", [5])]),
    (2, "multihop", "From what age is adjunctive lamotrigine indicated for the listed epilepsy seizure types, and what does the label say about serious rash with valproate?",
     "Adjunctive treatment is indicated from age two years. Including valproate in a multidrug regimen increases the risk of serious, potentially life-threatening rash.",
     [("Adjunctive epilepsy age", [4]), ("Valproate and serious rash risk", [2, 15, 56])]),
    (3, "factual", "At what creatinine clearance is zoledronic acid injection contraindicated?",
     "It is contraindicated when creatinine clearance is below 35 mL/min or there is evidence of acute renal impairment.",
     [("Renal contraindication threshold", [17, 21, 25, 79, 95])]),
    (3, "multihop", "What renal test is required before each zoledronic acid injection dose, and what is the minimum infusion time?",
     "Obtain serum creatinine and calculate creatinine clearance before each dose. Infuse for no less than 15 minutes.",
     [("Pre-dose serum creatinine and clearance", [17]), ("Minimum infusion duration", [25])]),
    (4, "factual", "For which condition is glipizide indicated in this label?",
     "As an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.",
     [("Indicated population and condition", [7])]),
    (4, "multihop", "What does the glipizide label say about type 1 diabetes and hypoglycemia risk with renal or hepatic insufficiency?",
     "Glipizide is contraindicated in type 1 diabetes. Renal or hepatic insufficiency can increase the risk of serious hypoglycemic reactions.",
     [("Type 1 diabetes contraindication", [8]), ("Renal/hepatic insufficiency and hypoglycemia", [13, 14])]),
    (5, "factual", "For which disease stage is MATULANE indicated, and how is it used with other drugs?",
     "It is indicated in combination with other anticancer drugs for stage III or IV Hodgkin lymphoma.",
     [("Indication, stage and combination treatment", [1])]),
    (5, "multihop", "What does the MATULANE label say about alcohol and tyramine-containing foods during treatment?",
     "Avoid alcohol because a disulfiram-like reaction may occur; avoid tyramine-containing foods because they may cause sudden, severe high blood pressure.",
     [("Alcohol and disulfiram-like reaction", [12, 19, 38]), ("Tyramine and blood pressure", [20])]),
]


def build(pool: list[dict]) -> list[dict]:
    if tuple(row["set_id"] for row in pool[:6]) != EXPECTED_SET_IDS:
        raise ValueError("DailyMed pilot pool is reordered or has changed labels")
    records = []
    for n, (pool_index, task, question, answer, claims) in enumerate(ITEMS, 1):
        label = pool[pool_index]
        present = {section["section_index"] for section in label["sections"]}
        if any(not set(indices) <= present for _, indices in claims):
            raise ValueError(f"Missing SPL section for temporal-{n:04d}")
        records.append({
            "qid": f"temporal-{n:04d}", "set_id": label["set_id"],
            "spl_version": label["spl_version"], "published_date": label["published_date"],
            "source_url": label["source_url"], "source_sha256": label["source_sha256"],
            "task": task, "question": question, "reference_answer": answer,
            "claims": [{"claim": claim, "acceptable_section_indices": indices}
                       for claim, indices in claims],
            "answerable": True, "annotation_status": "assistant_draft_pending_independent_expert_review",
            "annotator_ids": [], "adjudication_notes": "",
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/snapshots/recent_label_pool_10.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/curated/temporal_pilot.jsonl"))
    args = parser.parse_args()
    pool = list(read_jsonl(args.pool))
    if len(pool) < 6:
        raise ValueError("Pilot requires the pinned pool of at least six labels")
    records = build(pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "questions": len(records),
                      "labels": len({record["set_id"] for record in records})}))


if __name__ == "__main__":
    main()
