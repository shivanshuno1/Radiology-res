# src/pipeline.py

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from template_editor import replace_section_content
from template_structure import build_template_structure
from gemini_extractor import extract_findings
from gemini_field_editor import edit_fields_batch
from impression_generator import generate_impression
from validator import (
    validate_pipeline_output,
    validate_untouched_sections,
)


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.CSV"
TEST_PATH = DATA_DIR / "test.csv"


# =========================================================
# Generic helpers
# =========================================================

def model_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert a Pydantic model or dictionary to a dictionary.
    """

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    raise TypeError(
        f"Unsupported object type: {type(obj)}"
    )


def normalize_name(name: str | None) -> str:
    """
    Normalize section names for matching.
    """

    if name is None:
        return ""

    return " ".join(
        str(name).upper().split()
    )


# =========================================================
# Template hierarchy helpers
# =========================================================

def find_section(
    sections: List[Dict[str, Any]],
    target_name: str,
    parent_name: str | None = None,
):
    """
    Recursively find a section.

    Parent-child matching is preferred when a parent is supplied.
    """

    target = normalize_name(target_name)
    parent = normalize_name(parent_name)

    # -----------------------------------------------------
    # Parent -> child search
    # -----------------------------------------------------

    if parent:

        for section in sections:

            section_name = normalize_name(
                section.get("name")
            )

            if section_name == parent:

                for child in section.get(
                    "children",
                    []
                ):

                    if (
                        normalize_name(
                            child.get("name")
                        )
                        == target
                    ):
                        return child

                result = find_section(
                    section.get(
                        "children",
                        []
                    ),
                    target_name,
                    parent_name,
                )

                if result is not None:
                    return result

        return None

    # -----------------------------------------------------
    # Target-only recursive search
    # -----------------------------------------------------

    for section in sections:

        if (
            normalize_name(
                section.get("name")
            )
            == target
        ):
            return section

        result = find_section(
            section.get(
                "children",
                []
            ),
            target_name,
            None,
        )

        if result is not None:
            return result

    return None


def find_other_findings(
    sections: List[Dict[str, Any]]
):
    """
    Locate OTHER FINDINGS recursively.
    """

    for section in sections:

        if (
            normalize_name(
                section.get("name")
            )
            == "OTHER FINDINGS"
        ):
            return section

        result = find_other_findings(
            section.get(
                "children",
                []
            )
        )

        if result is not None:
            return result

    return None


# =========================================================
# Step 1: Extract + validate
# =========================================================

def extract_and_validate(
    dictation: str,
    structure: Dict[str, Any],
):
    """
    Extract patient-specific findings.

    The current extractor performs deterministic validation internally
    through validate_result(), so the pipeline should not attempt to
    validate each finding again.
    """

    print("\n[1/4] Extracting findings...")

    extraction = extract_findings(
        dictation=dictation,
        structure=structure,
    )

    findings = [
        model_to_dict(finding)
        for finding in extraction.findings
    ]

    print(
        f"Extracted and validated "
        f"{len(findings)} findings."
    )

    invalid_findings = []

    return (
        findings,
        invalid_findings,
    )

# =========================================================
# Step 2: Group field modifications
# =========================================================

def group_findings_by_section(
    findings: List[Dict[str, Any]]
):
    """
    Group findings by target section.

    Multiple findings affecting the same template field
    are sent to ONE field-editor call.
    """

    grouped = {}
    insertions = []

    for finding in findings:

        action = finding.get(
            "action"
        )

        if action == "no_change":
            continue

        if action == "insert":

            insertions.append(
                finding
            )
            continue

        if action != "modify":
            continue

        target = finding.get(
            "target_section"
        )

        parent = finding.get(
            "parent_section"
        )

        if not target:
            continue

        key = (
            normalize_name(parent),
            normalize_name(target),
        )

        grouped.setdefault(
            key,
            [],
        ).append(finding)

    return (
        grouped,
        insertions,
    )


# =========================================================
# Contradiction context for insertion findings
# =========================================================

def get_insertion_contradiction_targets(
    finding: Dict[str, Any],
    structure: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Determine whether an inserted finding can contradict
    an existing statement in a related template field.

    IMPORTANT:
    The finding is NOT routed into the target field.

    It is supplied only as contradiction context so the field
    editor can remove directly contradictory template wording.
    """

    text = (
        finding.get("finding", "")
        .strip()
        .lower()
    )

    if not text:
        return []

    targets = []

    # -----------------------------------------------------
    # LABRUM
    # -----------------------------------------------------
    #
    # A labral abnormality may contradict:
    #
    #     "The labrum is intact."
    #
    # in GLENOHUMERAL JOINT.
    #
    # But the actual labral finding remains an insertion.
    # -----------------------------------------------------

    if "labrum" in text or "labral" in text:

        section = find_section(
            structure["findings"],
            "GLENOHUMERAL JOINT",
            "JOINTS",
        )

        if section is not None:

            targets.append(
                {
                    "section_name": (
                        "GLENOHUMERAL JOINT"
                    ),
                    "parent_section": "JOINTS",
                }
            )

    # -----------------------------------------------------
    # ROTATOR INTERVAL / CAPSULE
    # -----------------------------------------------------
    #
    # These findings can contradict normal capsular wording
    # in the glenohumeral joint field.
    # -----------------------------------------------------

    if (
        "rotator interval" in text
        or "glenohumeral capsule" in text
        or "inferior capsule" in text
        or "capsular thickening" in text
        or "capsular edema" in text
    ):

        section = find_section(
            structure["findings"],
            "GLENOHUMERAL JOINT",
            "JOINTS",
        )

        if section is not None:

            target = {
                "section_name": (
                    "GLENOHUMERAL JOINT"
                ),
                "parent_section": "JOINTS",
            }

            if target not in targets:
                targets.append(target)

    return targets

# =========================================================
# Step 2: Edit raw template
# =========================================================


def apply_field_edits_to_raw_template(
    template: str,
    findings: List[Dict[str, Any]],
):
    """
    Modify only fields explicitly affected by the dictation.

    All affected fields are sent to the LLM in ONE batch call.
    The original template remains the source of truth.
    """

    print("\n[2/4] Editing template fields...")

    print("\n========== FINDINGS BEFORE GROUPING ==========")
    for i, finding in enumerate(findings, 1):
        print(
            i,
            "| action:", finding.get("action"),
            "| target:", finding.get("target_section"),
            "| parent:", finding.get("parent_section"),
            "| finding:", finding.get("finding"),
        )
    print("==============================================\n")

    grouped, insertions = group_findings_by_section(findings)

    updated_template = template

    # Structure is kept from the ORIGINAL template so that
    # contradiction-context routing never depends on partially
    # edited content.
    original_structure = build_template_structure(
        template
    )

    print(
        f"Fields requiring edits: {len(grouped)}"
    )

    # ========================================================
    # BUILD ALL FIELD INPUTS FIRST
    # ========================================================

    fields_to_edit = []

    for (
        parent_name,
        target_name,
    ), field_findings in grouped.items():

        structure = build_template_structure(
            updated_template
        )

        section = find_section(
            structure["findings"],
            target_name,
            parent_name or None,
        )

        if section is None:

            print(
                "  WARNING: Could not find "
                f"{parent_name} -> {target_name}"
            )

            continue

        original_content = (
            section.get("content", "")
            .strip()
        )

        print(
            "  Queuing: "
            f"{parent_name + ' -> ' if parent_name else ''}"
            f"{target_name}"
        )

        # -------------------------------------------------
        # Find insertion findings that may contradict
        # statements inside this existing field.
        #
        # These findings are NOT incorporated into the
        # field. They are supplied only as contradiction
        # context.
        # -------------------------------------------------

        contradiction_context = []

        for insertion in insertions:

            contradiction_targets = (
                get_insertion_contradiction_targets(
                    finding=insertion,
                    structure=original_structure,
                )
            )

            current_target = {
                "section_name": target_name,
                "parent_section": (
                    parent_name
                    if parent_name
                    else None
                ),
            }

            if current_target in contradiction_targets:

                contradiction_context.append(
                    {
                        "finding": insertion.get(
                            "finding",
                            "",
                        ),
                        "negated": insertion.get(
                            "negated",
                            False,
                        ),
                    }
                )

        fields_to_edit.append(
            {
                "section_name": target_name,
                "parent_section": (
                    parent_name
                    if parent_name
                    else None
                ),
                "original_content": original_content,
                "findings": field_findings,
                "contradiction_context": (
                    contradiction_context
                ),
            }
        )

    # ========================================================
    # ONE LLM CALL FOR ALL FIELDS
    # ========================================================

    if not fields_to_edit:

        print("  No fields require LLM editing.")

        return (
            updated_template,
            insertions,
        )

    print(
        f"\n  Sending {len(fields_to_edit)} fields "
        "in ONE batch call..."
    )

    edited_fields = edit_fields_batch(
        fields_to_edit,
        all_findings=findings,
    )

    print(
        f"  Batch editor returned "
        f"{len(edited_fields)} fields."
    )

    # ========================================================
    # APPLY EDITS DETERMINISTICALLY
    # ========================================================

    for result in edited_fields:

        target_name = result["section_name"]
        parent_name = result.get("parent_section")

        if not result["changed"]:

            print(
                "  No change:",
                (
                    f"{parent_name} -> {target_name}"
                    if parent_name
                    else target_name
                ),
            )

            continue

        edited_content = (
            result["edited_content"]
            .strip()
        )

        if not edited_content:

            print(
                "  WARNING: Empty edited content for "
                f"{target_name}. Keeping original."
            )

            continue

        print(
            "  Applying:",
            (
                f"{parent_name} -> {target_name}"
                if parent_name
                else target_name
            ),
        )

        updated_template = (
            replace_section_content(
                template=updated_template,
                target_name=target_name,
                parent_name=(
                    parent_name
                    if parent_name
                    else None
                ),
                new_content=edited_content,
            )
        )

    return (
        updated_template,
        insertions,
    )



# =========================================================
# Step 3: Apply insertions
# =========================================================

def apply_insertions_to_raw_template(
    template: str,
    insertions: List[Dict[str, Any]],
):
    """
    Add genuinely unrouted findings.

    Preferred behavior:
        1. Add to OTHER FINDINGS if present.
        2. Otherwise insert before IMPRESSION.

    We never create a fake anatomical section.
    """

    if not insertions:
        return template

    print(
        f"\nApplying "
        f"{len(insertions)} insertion(s)..."
    )

    texts = []

    for finding in insertions:

        text = (
            finding.get(
                "finding",
                "",
            )
            .strip()
        )

        if text:
            texts.append(text)

    if not texts:
        return template

    additions = "\n".join(
        texts
    )

    structure = build_template_structure(
        template
    )

    other = find_other_findings(
        structure["findings"]
    )

    # -----------------------------------------------------
    # OTHER FINDINGS exists
    # -----------------------------------------------------

    if other is not None:

        print(
            "  Adding insertions to "
            "OTHER FINDINGS."
        )

        existing = (
            other.get(
                "content",
                "",
            )
            .strip()
        )

        if existing:

            new_content = (
                existing
                + "\n"
                + additions
            )

        else:

            new_content = additions

        return replace_section_content(
            template=template,
            target_name="OTHER FINDINGS",
            parent_name=None,
            new_content=new_content,
        )

    # -----------------------------------------------------
    # No OTHER FINDINGS
    # -----------------------------------------------------

    print(
        "  No OTHER FINDINGS section found."
    )

    print(
        "  Creating OTHER FINDINGS section "
        "before IMPRESSION."
    )

    marker = "IMPRESSION:"

    index = template.upper().find(
        marker
    )

    if index == -1:

        print(
            "  WARNING: IMPRESSION marker "
            "not found. Insertions skipped."
        )

        return template

    before = template[:index].rstrip()
    after = template[index:].lstrip()

    other_findings_block = (
        "OTHER FINDINGS:\n"
        + additions
    )

    if before:
        before += "\n\n"

    return (
        before
        + other_findings_block
        + "\n\n"
        + after
    )

# =========================================================
# Step 4: Assemble final report
# =========================================================

def assemble_report(
    findings_template: str,
    impression: str,
):
    """
    Replace the template's original IMPRESSION with
    the generated impression.

    Everything before IMPRESSION is preserved.
    """

    marker = "IMPRESSION:"

    index = (
        findings_template
        .upper()
        .find(marker)
    )

    if index == -1:

        raise ValueError(
            "Template does not contain "
            "'IMPRESSION:' marker."
        )

    findings_part = (
        findings_template[:index]
        .rstrip()
    )

    clean_impression = (
        impression
        .strip()
    )

    if not clean_impression:

        raise ValueError(
            "Generated impression is empty."
        )

    report = (
        findings_part
        + "\n\n"
        + "IMPRESSION:\n"
        + clean_impression
    )

    return report


# =========================================================
# Complete case pipeline
# =========================================================

def run_case(
    template: str,
    dictation: str,
):
    """
    Run one complete radiology reporting case.

    Pipeline:

        Template
            ↓
        Structure
            ↓
        Extract
            ↓
        Validate
            ↓
        Route
            ↓
        Edit affected fields
            ↓
        Apply insertions
            ↓
        Generate impression
            ↓
        Assemble final report
            ↓
        Validate
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RUNNING RADIOLOGY REPORTING PIPELINE"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # Build structure
    # -----------------------------------------------------

    print(
        "\nBuilding template structure..."
    )

    structure = build_template_structure(
        template
    )

    # -----------------------------------------------------
    # Extract + validate
    # -----------------------------------------------------

    (
        valid_findings,
        invalid_findings,
    ) = extract_and_validate(
        dictation=dictation,
        structure=structure,
    )

    # -----------------------------------------------------
    # Edit raw template
    # -----------------------------------------------------

    (
        updated_template,
        insertions,
    ) = apply_field_edits_to_raw_template(
        template=template,
        findings=valid_findings,
    )

    # -----------------------------------------------------
    # Apply insertions
    # -----------------------------------------------------

    updated_template = (
        apply_insertions_to_raw_template(
            template=updated_template,
            insertions=insertions,
        )
    )

    # -----------------------------------------------------
    # Generate impression
    # -----------------------------------------------------

    print(
        "\n[3/4] Generating impression..."
    )

    impression_findings = [
        finding
        for finding in valid_findings
        if finding.get(
            "action"
        )
        != "no_change"
    ]

    impression = generate_impression(
        impression_findings
    )

    # -----------------------------------------------------
    # Assemble report
    # -----------------------------------------------------

    print(
        "\n[4/4] Rendering final report..."
    )

    report = assemble_report(
        findings_template=updated_template,
        impression=impression,
    )

    # -----------------------------------------------------
    # Validate findings + final report
    # -----------------------------------------------------

    validation = (
        validate_pipeline_output(
            findings=valid_findings,
            template_sections=(
                structure["findings"]
            ),
            report=report,
        )
    )

    # -----------------------------------------------------
    # Validate untouched template sections
    # -----------------------------------------------------

    original_sections = structure["findings"]

    updated_structure = build_template_structure(
        updated_template
    )

    updated_sections = updated_structure["findings"]

    edited_fields = []

    for finding in valid_findings:

        if finding.get("action") != "modify":
            continue

        target = finding.get("target_section")

        if not target:
            continue

        edited_fields.append(
            {
                "section_name": target,
                "parent_section": (
                    finding.get("parent_section")
                ),
            }
        )

    preservation_validation = (
        validate_untouched_sections(
            original_sections=original_sections,
            updated_sections=updated_sections,
            edited_fields=edited_fields,
        )
    )

    validation["template_preservation"] = (
        preservation_validation
    )

    validation["errors"].extend(
        preservation_validation["errors"]
    )

    validation["warnings"].extend(
        preservation_validation["warnings"]
    )

    if preservation_validation["errors"]:
        validation["valid"] = False

    # -----------------------------------------------------
    # Print final validation result
    # -----------------------------------------------------

    if not validation["valid"]:

        print(
            "\nWARNING: Final validation failed."
        )

        for error in validation["errors"]:

            print(
                f"  ❌ {error}"
            )

    else:

        print(
            "\nFinal validation: PASSED"
        )

    return {
        "report": report,
        "findings": valid_findings,
        "invalid_findings": (
            invalid_findings
        ),
        "impression": impression,
        "validation": validation,
    }

# =========================================================
# Generate final submission.csv for all test cases
# =========================================================

# =========================================================
# Generate submission with checkpoint + resume
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GENERATING FINAL SUBMISSION")
    print("=" * 70)

    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------

    CHECKPOINT_PATH = (
        PROJECT_ROOT / "submission_checkpoint.csv"
    )

    FAILED_PATH = (
        PROJECT_ROOT / "failed_cases.csv"
    )

    SUBMISSION_PATH = (
        PROJECT_ROOT / "submission.csv"
    )

    # -----------------------------------------------------
    # Load training data
    # -----------------------------------------------------

    print("\nLoading training data from:")
    print(TRAIN_PATH)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{TRAIN_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH)

    print(
        f"Loaded {len(train)} training cases."
    )

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    print("\nLoading test data from:")
    print(TEST_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Could not find:\n{TEST_PATH}"
        )

    test = pd.read_csv(TEST_PATH)

    print(
        f"Loaded {len(test)} test cases."
    )

    total = len(test)

    if total != 132:

        print(
            f"WARNING: Expected 132 test cases, "
            f"but found {total}."
        )

    # =====================================================
    # LOAD EXISTING CHECKPOINT
    # =====================================================

    submissions = []

    if CHECKPOINT_PATH.exists():

        print("\n" + "=" * 70)
        print("CHECKPOINT FOUND")
        print("=" * 70)

        try:

            checkpoint = pd.read_csv(
                CHECKPOINT_PATH
            )

            required_columns = [
                "case_id",
                "report",
            ]

            if checkpoint.columns.tolist() != required_columns:

                print(
                    "WARNING: Invalid checkpoint columns."
                )

                print(
                    "Ignoring checkpoint."
                )

            else:

                # Remove accidental duplicates
                checkpoint = (
                    checkpoint
                    .drop_duplicates(
                        subset=["case_id"],
                        keep="last",
                    )
                )

                # Remove empty reports
                checkpoint = checkpoint[
                    checkpoint["report"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                ]

                submissions = (
                    checkpoint[
                        ["case_id", "report"]
                    ]
                    .to_dict(
                        orient="records"
                    )
                )

                print(
                    f"Loaded "
                    f"{len(submissions)} "
                    f"successful cases "
                    f"from checkpoint."
                )

        except Exception as e:

            print(
                "WARNING: Could not read checkpoint."
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print(
                "Starting with empty checkpoint."
            )

            submissions = []

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting from case 1."
        )

    # =====================================================
    # COMPLETED CASE IDS
    # =====================================================

    completed_ids = {
        str(item["case_id"])
        for item in submissions
    }

    print(
        f"\nAlready completed: "
        f"{len(completed_ids)}/{total}"
    )

    # =====================================================
    # LOAD FAILED CASES
    # =====================================================

    failed_cases = {}

    if FAILED_PATH.exists():

        try:

            failed_df = pd.read_csv(
                FAILED_PATH
            )

            for _, failed_row in failed_df.iterrows():

                failed_cases[
                    str(failed_row["case_id"])
                ] = {
                    "case_id": str(
                        failed_row["case_id"]
                    ),
                    "error_type": str(
                        failed_row.get(
                            "error_type",
                            "",
                        )
                    ),
                    "error": str(
                        failed_row.get(
                            "error",
                            "",
                        )
                    ),
                }

            print(
                f"Previous failed cases: "
                f"{len(failed_cases)}"
            )

        except Exception as e:

            print(
                "WARNING: Could not read "
                "failed_cases.csv."
            )

    # =====================================================
    # CHECKPOINT SAVE FUNCTION
    # =====================================================

    def save_checkpoint():

        checkpoint_df = pd.DataFrame(
            submissions,
            columns=[
                "case_id",
                "report",
            ],
        )

        checkpoint_df = (
            checkpoint_df
            .drop_duplicates(
                subset=["case_id"],
                keep="last",
            )
        )

        # -------------------------------------------------
        # Atomic write
        # -------------------------------------------------

        temp_path = (
            PROJECT_ROOT
            / "submission_checkpoint.tmp.csv"
        )

        checkpoint_df.to_csv(
            temp_path,
            index=False,
            encoding="utf-8",
        )

        temp_path.replace(
            CHECKPOINT_PATH
        )

    # =====================================================
    # FAILED CASE SAVE FUNCTION
    # =====================================================

    def save_failed_cases():

        if not failed_cases:

            return

        failed_df = pd.DataFrame(
            list(
                failed_cases.values()
            ),
            columns=[
                "case_id",
                "error_type",
                "error",
            ],
        )

        failed_df.to_csv(
            FAILED_PATH,
            index=False,
            encoding="utf-8",
        )

    # =====================================================
    # PROCESS TEST CASES
    # =====================================================

    for i, (_, row) in enumerate(
        test.iterrows(),
        start=1,
    ):

        case_id = str(
            row["case_id"]
        )

        # -------------------------------------------------
        # SKIP ALREADY COMPLETED CASES
        # -------------------------------------------------

        if case_id in completed_ids:

            print(
                f"\n[{i}/{total}] "
                f"SKIPPING ALREADY COMPLETED"
            )

            print(
                f"case_id: {case_id}"
            )

            continue

        # -------------------------------------------------
        # CASE HEADER
        # -------------------------------------------------

        print("\n")
        print("=" * 70)

        print(
            f"PROCESSING TEST CASE "
            f"{i}/{total}"
        )

        print(
            f"case_id: {case_id}"
        )

        print(
            f"modality: "
            f"{row['modality']}"
        )

        print(
            f"body_part: "
            f"{row['body_part']}"
        )

        print("=" * 70)

        # =================================================
        # RUN CASE
        # =================================================

        try:

            result = run_case(
                template=row[
                    "template_content"
                ],
                dictation=row[
                    "dictation"
                ],
            )

            report = result["report"]

            # -------------------------------------------------
            # Basic safety check
            # -------------------------------------------------

            if (
                not report
                or not str(report).strip()
            ):

                raise ValueError(
                    "Pipeline returned "
                    "an empty report."
                )

            # -------------------------------------------------
            # Store successful case
            # -------------------------------------------------

            submissions.append(
                {
                    "case_id": case_id,
                    "report": str(report),
                }
            )

            completed_ids.add(
                case_id
            )

            # -------------------------------------------------
            # If previously failed, remove failure
            # -------------------------------------------------

            failed_cases.pop(
                case_id,
                None,
            )

            # -------------------------------------------------
            # SAVE IMMEDIATELY
            # -------------------------------------------------

            save_checkpoint()
            save_failed_cases()

            print(
                f"\nSUCCESS: "
                f"{len(completed_ids)}/{total}"
            )

            print(
                f"Checkpoint saved → "
                f"{CHECKPOINT_PATH}"
            )

        # =================================================
        # ANY ERROR
        # =================================================

        except Exception as e:

            error_type = (
                type(e).__name__
            )

            error_message = str(e)

            print("\n")
            print("!" * 70)

            print(
                f"ERROR processing case "
                f"{case_id}"
            )

            print(
                f"{error_type}: "
                f"{error_message}"
            )

            print(
                "CASE WILL BE SKIPPED."
            )

            print(
                "Continuing with next case..."
            )

            print("!" * 70)

            # -------------------------------------------------
            # Record failure
            # -------------------------------------------------

            failed_cases[
                case_id
            ] = {
                "case_id": case_id,
                "error_type": error_type,
                "error": error_message,
            }

            # -------------------------------------------------
            # SAVE BOTH FILES
            # -------------------------------------------------

            save_checkpoint()
            save_failed_cases()

            # -------------------------------------------------
            # DO NOT RAISE
            # -------------------------------------------------

            continue

    # =====================================================
    # FINAL STATUS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("BATCH PROCESSING FINISHED")
    print("=" * 70)

    print(
        f"Successful: "
        f"{len(completed_ids)}/{total}"
    )

    print(
        f"Failed: "
        f"{len(failed_cases)}"
    )

    # =====================================================
    # SAVE FINAL CHECKPOINT
    # =====================================================

    save_checkpoint()
    save_failed_cases()

    # =====================================================
    # IF SOME CASES FAILED
    # =====================================================

    if len(completed_ids) != total:

        missing_ids = [
            str(case_id)
            for case_id in test["case_id"]
            if str(case_id)
            not in completed_ids
        ]

        print("\n")
        print("=" * 70)
        print("INCOMPLETE SUBMISSION")
        print("=" * 70)

        print(
            f"\nCompleted: "
            f"{len(completed_ids)}/{total}"
        )

        print(
            f"Remaining: "
            f"{len(missing_ids)}"
        )

        print(
            f"\nCheckpoint:"
            f"\n{CHECKPOINT_PATH}"
        )

        print(
            f"\nFailed cases:"
            f"\n{FAILED_PATH}"
        )

        print(
            "\nThe program did NOT create "
            "a final submission.csv because "
            "some cases are missing."
        )

        print(
            "\nFix the failed cases/API issue "
            "and run the pipeline again."
        )

        print(
            "It will automatically skip "
            "successful cases."
        )

        raise SystemExit(1)

    # =====================================================
    # ALL CASES SUCCESSFUL
    # =====================================================

    submission = pd.DataFrame(
        submissions,
        columns=[
            "case_id",
            "report",
        ],
    )

    # -----------------------------------------------------
    # Ensure exact test order
    # -----------------------------------------------------

    test_order = {
        str(case_id): index
        for index, case_id
        in enumerate(
            test["case_id"]
        )
    }

    submission["_order"] = (
        submission["case_id"]
        .astype(str)
        .map(test_order)
    )

    submission = (
        submission
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("VALIDATING FINAL SUBMISSION")
    print("=" * 70)

    if len(submission) != total:

        raise ValueError(
            f"Expected {total} submission rows, "
            f"got {len(submission)}."
        )

    if submission.columns.tolist() != [
        "case_id",
        "report",
    ]:

        raise ValueError(
            "Submission columns are incorrect."
        )

    if (
        submission["case_id"]
        .nunique()
        != total
    ):

        raise ValueError(
            "Duplicate or missing "
            "case IDs detected."
        )

    if submission["report"].isna().any():

        raise ValueError(
            "Some reports are missing."
        )

    if (
        submission["report"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        raise ValueError(
            "Some reports are empty."
        )

    # =====================================================
    # SAVE FINAL SUBMISSION
    # =====================================================

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n")
    print("=" * 70)
    print("SUBMISSION CREATED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nFile:"
        f"\n{SUBMISSION_PATH}"
    )

    print(
        f"\nRows: "
        f"{len(submission)}"
    )

    print(
        f"Unique case IDs: "
        f"{submission['case_id'].nunique()}"
    )

    print(
        f"Columns: "
        f"{submission.columns.tolist()}"
    )

    print("\nFirst 5 rows:")

    print(
        submission.head()
    )

    print("\nDone.")