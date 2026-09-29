"""Select high-quality clinical simulation cases from Hugging Face.

The script streams AGBonnet/augmented-clinical-notes, rejects records that
cannot support the simulation, scores the remaining records, applies a simple
clinical-group diversity cap, and writes the selected records as JSONL.

It uses only the standard library and httpx, which is already a project
dependency. No records are reviewed or labelled by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes/"
    "resolve/main/augmented_notes_30K.jsonl"
)
SELECTION_VERSION = "1.0"
EMPTY_VALUES = {"", "none", "null", "n/a", "na", "unknown", "not available"}
ROLE_PATTERN = re.compile(r"(?i)(?:^|\n|\s)(?:\d+\.\s*)?(doctor|patient):\s*")
WORD_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}

PROMPT_LEAK_MARKERS = (
    "as an ai language model",
    "ignore previous instructions",
    "given the clinical note",
    "generate a conversation between",
    "the following conversation",
    "your task is to generate",
    "system prompt",
)

CLINICAL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cardiovascular",
        (
            "aorta",
            "artery",
            "cardiac",
            "cardio",
            "coronary",
            "heart",
            "myocard",
            "vascular",
        ),
    ),
    (
        "respiratory",
        (
            "asthma",
            "bronch",
            "dyspnea",
            "lung",
            "pleural",
            "pneum",
            "pulmonary",
            "respirat",
        ),
    ),
    (
        "gastrointestinal",
        (
            "abdomen",
            "abdominal",
            "bowel",
            "colon",
            "gastric",
            "gastro",
            "intestinal",
            "liver",
            "pancrea",
        ),
    ),
    (
        "neurology",
        (
            "brain",
            "cerebral",
            "headache",
            "neurolog",
            "seizure",
            "spinal",
            "stroke",
        ),
    ),
    (
        "musculoskeletal",
        (
            "arthritis",
            "bone",
            "fracture",
            "hip",
            "joint",
            "knee",
            "muscle",
            "orthop",
            "spine",
        ),
    ),
    (
        "endocrine_metabolic",
        (
            "adrenal",
            "diabet",
            "endocr",
            "metabolic",
            "osteomalacia",
            "pituitary",
            "thyroid",
        ),
    ),
    (
        "renal_urology",
        (
            "bladder",
            "hematuria",
            "kidney",
            "renal",
            "ureter",
            "urinary",
            "urolog",
        ),
    ),
    (
        "haematology_oncology",
        (
            "cancer",
            "carcinoma",
            "hematolog",
            "leukemia",
            "lymphoma",
            "malignan",
            "oncolog",
            "tumor",
            "tumour",
        ),
    ),
    (
        "infectious_disease",
        (
            "bacterial",
            "fever",
            "fungal",
            "infection",
            "tuberculosis",
            "viral",
        ),
    ),
    (
        "dermatology",
        ("dermat", "lesion", "rash", "skin"),
    ),
    (
        "mental_health",
        ("anxiety", "depression", "psychiatr", "psycholog", "psychosis"),
    ),
    (
        "obstetrics_gynaecology",
        (
            "cervical",
            "gynaec",
            "gynec",
            "ovarian",
            "pregnan",
            "uterine",
            "uterus",
        ),
    ),
    (
        "ear_nose_throat",
        ("ear", "hearing", "laryn", "nasal", "sinus", "throat", "tinnitus"),
    ),
    (
        "ophthalmology",
        ("eye", "ocular", "ophthalm", "retina", "vision"),
    ),
)


@dataclass(slots=True)
class Candidate:
    row: dict[str, Any]
    score: int
    group: str
    reasons: list[str]
    metrics: dict[str, Any]
    tie_breaker: int


def text(value: Any) -> str:
    """Return a stripped string for a dataset value."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


def is_present(value: Any) -> bool:
    """Return True when a scalar or nested value contains useful content."""
    if isinstance(value, str):
        return value.strip().lower() not in EMPTY_VALUES
    if isinstance(value, dict):
        return any(is_present(item) for item in value.values())
    if isinstance(value, list):
        return any(is_present(item) for item in value)
    return value is not None


def count_present_fields(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(1 for item in value.values() if is_present(item))


def parse_summary(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def extract_diagnoses(summary: dict[str, Any]) -> list[str]:
    diagnoses: list[str] = []
    for item in list_of_dicts(summary.get("diagnosis tests")):
        condition = text(item.get("condition"))
        if is_present(condition) and condition not in diagnoses:
            diagnoses.append(condition)
    return diagnoses


def extract_symptoms(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in list_of_dicts(summary.get("symptoms"))
        if is_present(item.get("name of symptom"))
    ]


def conversation_parts(conversation: str) -> list[tuple[str, str]]:
    matches = list(ROLE_PATTERN.finditer(conversation))
    parts: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(conversation)
        )
        content = conversation[match.end() : end].strip()
        if content:
            parts.append((match.group(1).lower(), content))
    return parts


def normalized_words(value: str) -> list[str]:
    return [
        word for word in WORD_PATTERN.findall(value.lower()) if word not in STOPWORDS
    ]


def normalized_phrase(value: str) -> str:
    return " ".join(normalized_words(value))


def diagnosis_is_supported(diagnosis: str, full_note: str) -> bool:
    diagnosis_words = set(normalized_words(diagnosis))
    if not diagnosis_words:
        return False
    note_words = set(normalized_words(full_note))
    overlap = len(diagnosis_words & note_words) / len(diagnosis_words)
    return overlap >= 0.6


def diagnosis_is_revealed(diagnosis: str, opening_text: str) -> bool:
    normalized_diagnosis = normalized_phrase(diagnosis)
    normalized_opening = normalized_phrase(opening_text)
    return bool(
        normalized_diagnosis
        and len(normalized_diagnosis) >= 5
        and normalized_diagnosis in normalized_opening
    )


def clinical_group(summary: dict[str, Any], diagnoses: list[str]) -> str:
    source = " ".join([text(summary.get("visit motivation")), *diagnoses]).lower()
    for group, keywords in CLINICAL_GROUPS:
        if any(keyword in source for keyword in keywords):
            return group
    return "other"


def stable_tie_breaker(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def evaluate_record(
    row: dict[str, Any], seed: int
) -> tuple[Candidate | None, str | None]:
    case_id = text(row.get("idx"))
    full_note = text(row.get("full_note"))
    conversation = text(row.get("conversation"))
    summary = parse_summary(row.get("summary"))

    if not case_id:
        return None, "missing_case_id"
    if len(full_note) < 1_000:
        return None, "full_note_too_short"
    if len(conversation) < 500:
        return None, "conversation_too_short"
    if summary is None:
        return None, "invalid_summary_json"

    complaint = text(summary.get("visit motivation"))
    symptoms = extract_symptoms(summary)
    diagnoses = extract_diagnoses(summary)
    if not is_present(complaint):
        return None, "missing_complaint"
    if not symptoms:
        return None, "missing_symptoms"
    if not diagnoses:
        return None, "missing_diagnosis"

    lowered_conversation = conversation.lower()
    if any(marker in lowered_conversation for marker in PROMPT_LEAK_MARKERS):
        return None, "prompt_leak"

    parts = conversation_parts(conversation)
    doctor_parts = [content for role, content in parts if role == "doctor"]
    patient_parts = [content for role, content in parts if role == "patient"]
    if len(doctor_parts) < 2 or len(patient_parts) < 2:
        return None, "too_few_dialogue_turns"

    meaningful_patient_parts = [
        reply for reply in patient_parts if len(normalized_words(reply)) >= 4
    ]
    meaningful_ratio = len(meaningful_patient_parts) / len(patient_parts)
    if meaningful_ratio < 0.4:
        return None, "low_quality_patient_responses"

    first_patient_reply = patient_parts[0] if patient_parts else ""
    opening_text = f"{complaint} {first_patient_reply}"
    if any(diagnosis_is_revealed(item, opening_text) for item in diagnoses):
        return None, "diagnosis_revealed_in_opening"

    supported_diagnoses = [
        item for item in diagnoses if diagnosis_is_supported(item, full_note)
    ]
    if not supported_diagnoses:
        return None, "diagnosis_not_supported_by_full_note"

    score = 0
    reasons: list[str] = []

    score += 10
    reasons.append("clear presenting complaint")

    symptom_points = min(15, 5 + (2 * len(symptoms)))
    score += symptom_points
    if len(symptoms) >= 3:
        reasons.append("multiple recorded symptoms")

    history_fields = count_present_fields(summary.get("patient medical history"))
    history_points = min(10, history_fields * 2)
    score += history_points
    if history_fields >= 2:
        reasons.append("useful medical history")

    patient_fields = count_present_fields(summary.get("patient information"))
    score += min(5, patient_fields)

    minimum_turns = min(len(doctor_parts), len(patient_parts))
    dialogue_points = 8
    if minimum_turns >= 5:
        dialogue_points += 5
    if minimum_turns >= 8:
        dialogue_points += 3
    dialogue_points += round(9 * meaningful_ratio)
    score += min(25, dialogue_points)
    if minimum_turns >= 5 and meaningful_ratio >= 0.7:
        reasons.append("substantial patient conversation")

    if len(diagnoses) == 1:
        score += 10
        reasons.append("single clear recorded diagnosis")
    elif len(diagnoses) == 2:
        score += 7
    else:
        score += 4
    score += 10
    reasons.append("diagnosis supported by full note")

    if len(full_note) >= 2_000:
        score += 5
    if len(full_note) >= 4_000:
        score += 5
        reasons.append("detailed source note")

    examinations = list_of_dicts(summary.get("medical examinations"))
    if any(is_present(item) for item in examinations):
        score += 3
    diagnosis_tests = list_of_dicts(summary.get("diagnosis tests"))
    if any(is_present(item.get("result")) for item in diagnosis_tests):
        score += 2

    score = min(100, score)
    group = clinical_group(summary, diagnoses)
    metrics = {
        "diagnosis_count": len(diagnoses),
        "doctor_turns": len(doctor_parts),
        "full_note_characters": len(full_note),
        "meaningful_patient_response_ratio": round(meaningful_ratio, 3),
        "patient_turns": len(patient_parts),
        "symptom_count": len(symptoms),
    }
    return (
        Candidate(
            row=row,
            score=score,
            group=group,
            reasons=reasons,
            metrics=metrics,
            tie_breaker=stable_tie_breaker(case_id, seed),
        ),
        None,
    )


def iter_local_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object on line {line_number}")  # noqa: TRY004
            yield row


def iter_remote_rows(url: str) -> Iterator[dict[str, Any]]:
    timeout = httpx.Timeout(120.0, read=None)
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        for line_number, line in enumerate(response.iter_lines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object on line {line_number}")  # noqa: TRY004
            yield row


def build_candidate_pool(
    rows: Iterable[dict[str, Any]],
    *,
    pool_size: int,
    seed: int,
) -> tuple[list[Candidate], Counter[str], int]:
    heap: list[tuple[int, int, int, Candidate]] = []
    rejected: Counter[str] = Counter()
    total = 0

    for sequence, row in enumerate(rows):
        total += 1
        candidate, reason = evaluate_record(row, seed)
        if candidate is None:
            rejected[reason or "unknown"] += 1
            continue

        item = (candidate.score, candidate.tie_breaker, sequence, candidate)
        if len(heap) < pool_size:
            heapq.heappush(heap, item)
        elif item[:3] > heap[0][:3]:
            heapq.heapreplace(heap, item)

        if total % 5_000 == 0:
            print(
                f"Scanned {total:,} records; retained {len(heap):,} candidates",
                file=sys.stderr,
            )

    candidates = [item[3] for item in heap]
    candidates.sort(
        key=lambda item: (-item.score, item.tie_breaker, text(item.row.get("idx")))
    )
    return candidates, rejected, total


def select_diverse_cases(
    candidates: list[Candidate],
    *,
    count: int,
    max_group_share: float,
) -> list[Candidate]:
    max_per_group = max(1, math.ceil(count * max_group_share))
    group_counts: Counter[str] = Counter()
    selected: list[Candidate] = []
    deferred: list[Candidate] = []

    for candidate in candidates:
        if group_counts[candidate.group] >= max_per_group:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        group_counts[candidate.group] += 1
        if len(selected) == count:
            return selected

    # If the diversity cap prevents a complete selection, fill the remainder
    # with the highest-scoring deferred candidates.
    for candidate in deferred:
        selected.append(candidate)
        if len(selected) == count:
            return selected
    return selected


def write_jsonl(
    output: Path,
    selected: list[Candidate],
    *,
    seed: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for rank, candidate in enumerate(selected, start=1):
                record = dict(candidate.row)
                record["_selection"] = {
                    "clinical_group": candidate.group,
                    "metrics": candidate.metrics,
                    "quality_score": candidate.score,
                    "rank": rank,
                    "reasons": candidate.reasons,
                    "seed": seed,
                    "version": SELECTION_VERSION,
                }
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and select high-quality cases from AGBonnet/augmented-clinical-notes"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clinical_cases_200.jsonl"),
        help="Output JSONL path (default: data/clinical_cases_200.jsonl)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of cases to select (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic tie-breaking (default: 42)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Read a local JSONL file instead of downloading the dataset",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_DATASET_URL,
        help="Hugging Face JSONL URL",
    )
    parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=10,
        help="Keep count × this many top candidates before diversity selection",
    )
    parser.add_argument(
        "--max-group-share",
        type=float,
        default=0.20,
        help="Maximum share from one clinical group before fallback (default: 0.20)",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.pool_multiplier < 1:
        parser.error("--pool-multiplier must be at least 1")
    if not 0 < args.max_group_share <= 1:
        parser.error("--max-group-share must be greater than 0 and at most 1")
    return args


def print_summary(
    *,
    output: Path,
    selected: list[Candidate],
    rejected: Counter[str],
    total: int,
) -> None:
    print(f"Scanned:  {total:,}")
    print(f"Selected: {len(selected):,}")
    print(f"Saved:    {output}")
    if selected:
        scores = [candidate.score for candidate in selected]
        print(f"Scores:   {min(scores)}-{max(scores)}")

    groups = Counter(candidate.group for candidate in selected)
    print("\nSelected clinical groups:")
    for group, amount in groups.most_common():
        print(f"  {group:<28} {amount:>4}")

    print("\nMost common rejection reasons:")
    for reason, amount in rejected.most_common(10):
        print(f"  {reason:<36} {amount:>6}")


def print_failure_summary(
    *,
    eligible: int,
    rejected: Counter[str],
    total: int,
) -> None:
    print(f"Scanned:  {total:,}", file=sys.stderr)
    print(f"Eligible: {eligible:,}", file=sys.stderr)
    print("\nMost common rejection reasons:", file=sys.stderr)
    for reason, amount in rejected.most_common(10):
        print(f"  {reason:<36} {amount:>6}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = iter_local_rows(args.input) if args.input else iter_remote_rows(args.url)
    pool_size = max(args.count, args.count * args.pool_multiplier)
    candidates, rejected, total = build_candidate_pool(
        rows,
        pool_size=pool_size,
        seed=args.seed,
    )
    selected = select_diverse_cases(
        candidates,
        count=args.count,
        max_group_share=args.max_group_share,
    )
    if len(selected) < args.count:
        print_failure_summary(
            eligible=len(candidates),
            rejected=rejected,
            total=total,
        )
        print(
            "\n"
            f"Only {len(selected)} records passed validation; "
            f"cannot select {args.count}.",
            file=sys.stderr,
        )
        return 1

    write_jsonl(args.output, selected, seed=args.seed)
    print_summary(
        output=args.output,
        selected=selected,
        rejected=rejected,
        total=total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
