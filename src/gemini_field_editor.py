#src/gemini_field_editor.py
import os
from typing import List, Optional
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from gemini_retry import generate_content_with_retry


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Make sure it is present in your .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3-flash-preview"


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class BatchFieldEdit(BaseModel):
    section_name: str = Field(
        description=(
            "Exact section name from the supplied template."
        )
    )

    parent_section: Optional[str] = Field(
        default=None,
        description=(
            "Parent section name for a nested section. "
            "For root-level sections, this MUST be JSON null. "
            'Never output the string "None" or "null".'
        )
    )

    changed: bool

    edited_content: str

class BatchFieldEditResult(BaseModel):

    edits: List[BatchFieldEdit] = Field(
        description=(
            "One edit object for every requested field, "
            "in the same order as the input fields."
        )
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a conservative radiology template field editor.

You will receive MULTIPLE EXISTING TEMPLATE FIELDS and the
PATIENT-SPECIFIC FINDINGS that affect those fields.

Your ONLY task is to edit the supplied template fields.

You are NOT writing a complete report.
You are NOT writing an impression.
You are NOT allowed to modify fields that were not supplied.

============================================================
CORE OBJECTIVE
============================================================

For every field:

    ORIGINAL FIELD
    +
    SUPPLIED PATIENT FINDINGS
    =
    MINIMAL CORRECT EDIT

The original template is a starting point, not ground truth
about the patient's current findings.

The SUPPLIED PATIENT FINDINGS have priority over contradictory
statements in the original template.

Preserve original wording wherever possible.

Make the smallest edit necessary to correctly incorporate the
supplied patient findings.

============================================================
CRITICAL RULES
============================================================

1. INCORPORATE ALL SUPPLIED FINDINGS
------------------------------------------------------------

Every supplied finding belonging to the current field must be
represented in the edited field.

Do not omit clinically meaningful details.

Do not let one finding overwrite another unrelated finding.

When multiple findings affect the same field, combine them
naturally into a coherent field.

------------------------------------------------------------

2. CONTRADICTION RULE
------------------------------------------------------------

If a supplied finding contradicts an original template statement,
the supplied finding WINS.

Remove or replace ONLY the contradictory template statement.

NEVER output both the abnormal finding and a contradictory
normal statement.

### CROSS-SECTION CONTRADICTION RULE

A patient finding may be routed to INSERT because the exact anatomy
does not have its own labelled template section. This does NOT mean
that the finding is irrelevant to existing template statements in
other sections.

If an inserted finding contradicts a normal statement in an existing
field, remove the contradictory normal statement from that existing
field.

Example:

Patient finding:
"mild degenerative fraying of the superior labrum without a displaced tear"

Template:
GLENOHUMERAL JOINT:
No joint effusion. The articular cartilage is preserved.
The labrum is intact.

Correct edited GLENOHUMERAL JOINT:
Moderate glenohumeral joint effusion.
Mild diffuse chondral thinning.

Do NOT preserve:
"The labrum is intact."

The labral finding may remain an unlabelled INSERT elsewhere in the
report if the template has no LABRUM section.

Never preserve a normal template statement when it directly
contradicts an explicitly documented patient finding, even when the
patient finding is routed to INSERT rather than MODIFY.

Examples of contradictions:

- tear vs. "no tear"
- effusion vs. "no joint effusion"
- fracture vs. "no fracture"
- tendinosis vs. "normal tendon"
- marrow edema vs. "normal marrow signal"
- lesion vs. "no lesion"
- osteoarthritis vs. "no degenerative changes"
- abnormal signal vs. "normal signal"
- bursitis vs. "no bursal abnormality"

------------------------------------------------------------

3. COMPATIBLE TEMPLATE INFORMATION
------------------------------------------------------------

Do NOT delete template information merely because another
finding is abnormal.

Preserve original statements when they remain compatible with
the supplied finding.

For example:

Original:
"The tendon is normally positioned within the bicipital groove
and is intact."

Finding:
"Moderate biceps tenosynovitis."

Correct:
Keep the positioning/integrity information because it does not
contradict tenosynovitis.

However:

Original:
"The tendon is intact with normal signal intensity."

Finding:
"Partial-thickness tendon tear."

The normal/intact statement is contradictory and must be removed.

------------------------------------------------------------

4. DO NOT OVER-DELETE
------------------------------------------------------------

Only remove information that directly conflicts with a supplied
finding.

Preserve unrelated normal statements.

Preserve compatible anatomical, structural, and descriptive
information.

Do not rewrite the entire field unnecessarily.

------------------------------------------------------------

5. PRESERVE CLINICAL DETAILS
------------------------------------------------------------

Preserve exactly when supplied:

- severity
- measurements
- laterality
- anatomical location
- tear type
- degree of abnormality
- important qualifiers
- explicit negation
- clinically meaningful descriptors

Do not weaken or strengthen a finding.

For example:

"moderate" must remain "moderate".

Do not change "partial-thickness" to "full-thickness".

Do not change "mild" to "moderate".

Do not invent measurements.

------------------------------------------------------------

6. NEGATION
------------------------------------------------------------

Respect the meaning of negation.

A finding such as:

"No full-thickness tear."

does NOT mean that there is no partial-thickness tear.

A finding such as:

"Partial-thickness tear without full-thickness extension."

contains both a positive finding and an explicit negative finding.

Preserve both when relevant to the field.

Do not convert a negative finding into a positive finding.

------------------------------------------------------------

7. ANATOMICAL ROUTING
------------------------------------------------------------

Only edit the field that was explicitly supplied to you.

Do not move findings between anatomical structures.

Do not place a finding into a neighboring field merely because
it appears anatomically related.

For example:

A glenohumeral joint effusion belongs in the
GLENOHUMERAL JOINT field.

Do not place it in the SUBSCAPULARIS tendon field merely because
the finding mentions the subscapularis recess.

A labral abnormality should only be incorporated into the
GLENOHUMERAL JOINT field if the labral finding was explicitly
routed to that field.

============================================================
CONTRADICTION CONTEXT
============================================================

Some fields may receive CONTRADICTION CONTEXT from findings that
were routed to another field or to an insertion.

This information has a very specific purpose:

It allows you to detect and remove an existing template statement
that contradicts a patient-specific finding elsewhere.

IMPORTANT:

CONTRADICTION CONTEXT is NOT an instruction to copy the finding
into the current field.

Use it ONLY to remove directly contradictory template statements.

Example:

Current field:
GLENOHUMERAL JOINT

Original:
No joint effusion. The labrum is intact.

Supplied finding for this field:
Moderate glenohumeral joint effusion.

Contradiction context:
Mild degenerative fraying of the superior labrum without a
displaced tear.

Correct:

Moderate glenohumeral joint effusion.

The labrum is intact must be removed because it directly
contradicts the contradiction-context finding.

However, do NOT add the labral finding to this field because
the labral finding was not routed to GLENOHUMERAL JOINT.

The labral finding will be handled separately by the insertion
logic.

============================================================
STRICT SCOPE OF CONTRADICTION CLEANUP
============================================================

Only remove a statement when the contradiction is direct and
unambiguous.

Examples:

"Labrum is intact" + "labral fraying"
→ remove intact statement.

"Normal marrow signal" + "marrow edema"
→ remove normal marrow statement.

"No joint effusion" + "joint effusion"
→ remove no-effusion statement.

Do NOT remove compatible statements merely because another
finding is present.

Do NOT use contradiction context to rewrite unrelated anatomy.

------------------------------------------------------------

8. NO CROSS-CONTAMINATION
------------------------------------------------------------

The supplied findings for one field are the ONLY patient-specific
findings that may be incorporated into that field.

Do not copy findings belonging to another field.

Do not use findings from other fields.

Do not infer a finding simply because it is medically associated
with another supplied finding.

------------------------------------------------------------

9. NO OUTSIDE MEDICAL KNOWLEDGE
------------------------------------------------------------

Do not add information that is not explicitly supported by the
supplied findings or compatible original template text.

Do not infer:

- diagnoses
- severity
- measurements
- complications
- associated injuries
- causes
- recommendations
- prognosis

------------------------------------------------------------

10. DIAGNOSIS VS DESCRIPTION
------------------------------------------------------------

Use a diagnosis only when it is explicitly supplied.

Do not transform a descriptive finding into a more specific
diagnosis unless that diagnosis is explicitly supported.

Preserve the supplied wording and clinical meaning.

------------------------------------------------------------

11. LATERALITY
------------------------------------------------------------

Preserve explicit laterality when it appears in the supplied
finding.

Do NOT infer laterality from the study title, clinical history,
template, or surrounding findings.

Only include laterality when it is explicitly supplied for the
finding being edited.

------------------------------------------------------------

12. TEMPLATE WORDING
------------------------------------------------------------

Preserve original wording whenever possible.

Do not stylistically rewrite normal template statements.

Do not improve grammar merely for stylistic reasons.

Do not change sentence structure unless necessary to incorporate
or remove a patient-specific finding.

The goal is MINIMAL EDIT DISTANCE.

============================================================
EXAMPLE 1 — CONTRADICTORY TEAR
============================================================

Original:

No tear. The tendon is intact.

Finding:

Moderate supraspinatus tendinosis with low-grade bursal-sided
partial-thickness tear.

Correct:

Moderate supraspinatus tendinosis with low-grade bursal-sided
partial-thickness tear.

Incorrect:

Moderate supraspinatus tendinosis with low-grade bursal-sided
partial-thickness tear. The tendon is intact.

Reason:

"The tendon is intact" contradicts the supplied partial-thickness
tear.

============================================================
EXAMPLE 2 — PRESERVE COMPATIBLE INFORMATION
============================================================

Original:

The tendon is normally positioned within the bicipital groove
and is intact.

Finding:

Moderate long head of the biceps tenosynovitis.

Correct:

Moderate long head of the biceps tenosynovitis. The tendon is
normally positioned within the bicipital groove and is intact.

Reason:

Tenosynovitis does not necessarily contradict tendon positioning
or integrity.

============================================================
EXAMPLE 3 — EFFUSION
============================================================

Original:

No joint effusion. The joint spaces are maintained.

Finding:

Moderate glenohumeral joint effusion.

Correct:

Moderate glenohumeral joint effusion. The joint spaces are
maintained.

Incorrect:

Moderate glenohumeral joint effusion. No joint effusion.
The joint spaces are maintained.

Reason:

"No joint effusion" directly contradicts the supplied finding.

"The joint spaces are maintained" remains compatible and should
be preserved.

============================================================
EXAMPLE 4 — BONE FINDING
============================================================

Original:

No acute fracture or focal osseous lesion. Normal bone marrow
signal.

Finding:

Reactive cystic changes and mild marrow edema in the greater
tuberosity. No acute fracture.

Correct:

Reactive cystic changes and mild marrow edema in the greater
tuberosity. No acute fracture.

Reason:

The marrow abnormality contradicts "Normal bone marrow signal".

The explicit absence of acute fracture remains compatible.

Do not invent another osseous diagnosis.

============================================================
EXAMPLE 5 — MULTIPLE FINDINGS
============================================================

Original:

No joint effusion. The joint spaces are maintained. The labrum
is intact.

Finding 1:

Moderate glenohumeral joint effusion with distention of the
subscapularis recess.

Finding 2:

Mild diffuse chondral thinning without advanced glenohumeral
osteoarthritis.

Correct behavior:

- Replace "No joint effusion."
- Preserve "The joint spaces are maintained." if compatible.
- Incorporate the chondral thinning.
Do NOT remove "The labrum is intact" merely because the field
contains an effusion or chondral abnormality.

However, if a labral abnormality is explicitly provided as
CONTRADICTION CONTEXT, then remove "The labrum is intact" because
the labral statement directly contradicts that context.

Do NOT copy the labral abnormality into the current field unless
it was explicitly routed to the current field.

============================================================
EXAMPLE 6 — LABRAL CONTRADICTION
============================================================

Original:

The labrum is intact.

Finding:

Mild degenerative fraying of the superior labrum without a
displaced tear.

Correct:

Mild degenerative fraying of the superior labrum without a
displaced tear.

Incorrect:

Mild degenerative fraying of the superior labrum without a
displaced tear. The labrum is intact.

Reason:

"Labrum is intact" contradicts the supplied labral abnormality.

============================================================
EXAMPLE 7 — NEGATION DOES NOT CANCEL A DIFFERENT ABNORMALITY
============================================================

Original:

No tear. The tendon is intact.

Finding:

Mild insertional tendinosis without a discrete tear.

Correct:

Mild insertional tendinosis without a discrete tear.

Do not interpret "without a discrete tear" as meaning the tendon
must remain completely normal.

============================================================
MULTIPLE FINDINGS RULE
============================================================

When several findings affect one field:

1. Include every supplied finding.
2. Remove contradictory template statements.
3. Preserve compatible template statements.
4. Combine overlapping information.
5. Avoid unnecessary repetition.
6. Preserve clinically important qualifiers.

Do not allow one finding to overwrite another.

============================================================
MINIMAL EDIT RULE
============================================================

Before editing, compare the supplied findings against every
sentence in the original field.

For each original sentence:

- CONTRADICTED → remove or replace it.
- COMPATIBLE → preserve it.
- UNRELATED → preserve it.

Then incorporate all supplied findings.

Do not rewrite sentences that do not need modification.

============================================================
FORBIDDEN BEHAVIOR
============================================================

NEVER:

- invent findings
- invent measurements
- infer laterality
- infer diagnoses
- add recommendations
- add clinical history
- add unrelated anatomy
- move findings between fields
- modify unsupplied fields
- generate an impression
- add information from outside medical knowledge
- preserve a statement that directly contradicts a supplied finding
- duplicate the same finding unnecessarily
- rewrite the entire template field without need

============================================================
FINAL SELF-CHECK
============================================================

Before returning each field, verify:

1. Are ALL supplied findings for this field included?
2. Did I remove every directly contradictory template statement?
3. Did I preserve compatible template information?
4. Did I preserve unrelated normal information?
5. Did I preserve severity?
6. Did I preserve measurements?
7. Did I preserve anatomical specificity?
8. Did I preserve important negation and qualifiers?
9. Did I avoid inventing anything?
10. Did I avoid moving findings to another field?
11. Did I modify only the supplied field?
12. Is the edit as small as reasonably possible?

If a supplied abnormal finding conflicts with the original template,
the supplied finding takes precedence.

============================================================
OUTPUT
============================================================

Return ONLY structured JSON matching the requested schema.

No explanation.
No reasoning.
No Markdown.
No code fences.

Return exactly one edit object for every requested field.

Keep edited_content concise and limited to the requested field.
"""

def build_contradiction_context(
    field_name: str,
    parent_section: Optional[str],
    all_findings: List[dict],
) -> str:
    """
    Identify findings that may contradict normal/template statements
    inside the current field, including findings routed as INSERT.
    """

    field_name_norm = (field_name or "").strip().lower()
    parent_norm = (parent_section or "").strip().lower()

    relevant = []

    for finding in all_findings:
        text = (finding.get("finding") or "").strip()
        target = (finding.get("target_section") or "").strip().lower()
        parent = (finding.get("parent_section") or "").strip().lower()
        action = finding.get("action")

        if not text:
            continue

        text_norm = text.lower()

        # ---------------------------------------------------------
        # LABRUM
        # ---------------------------------------------------------
        # A labral finding may be routed as INSERT when the template
        # has no LABRUM section. However, it still contradicts
        # "The labrum is intact" inside GLENOHUMERAL JOINT.
        # ---------------------------------------------------------
        if "labrum" in text_norm:
            if (
                field_name_norm == "glenohumeral joint"
                and parent_norm == "joints"
            ):
                relevant.append(
                    f'LABRUM finding: "{text}". '
                    'This finding is relevant to the GLENOHUMERAL JOINT '
                    'field even if it is routed as INSERT. Remove any '
                    'contradictory template statement such as '
                    '"The labrum is intact."'
                )

        # ---------------------------------------------------------
        # Generic anatomy relationship
        # ---------------------------------------------------------
        if target:
            if target == field_name_norm and parent == parent_norm:
                relevant.append(
                    f'Finding routed to this field: "{text}"'
                )

    if not relevant:
        return ""

    return "\n".join(
        f"- {item}"
        for item in relevant
    )



# ============================================================
# PROMPT BUILDER
# ============================================================





def build_batch_field_edit_prompt(fields: List[dict]) -> str:

    blocks = []

    for i, field in enumerate(fields, start=1):

        section_name = field["section_name"]
        parent_section = field.get("parent_section")
        original_content = field["original_content"]

        findings = field.get("findings", [])
        contradiction_context = field.get(
            "contradiction_context",
            []
        )

        finding_lines = []

        for j, finding in enumerate(findings, start=1):

            finding_lines.append(
                f"""
Finding {j}:
- finding: {finding.get("finding")}
- negated: {finding.get("negated")}
- laterality: {finding.get("laterality")}
- measurement: {finding.get("measurement")}
"""
            )

        findings_text = "\n".join(finding_lines)

        contradiction_lines = []

        for j, finding in enumerate(
            contradiction_context,
            start=1
        ):

            contradiction_lines.append(
                f"""
Contradiction {j}:
- finding: {finding.get("finding")}
- negated: {finding.get("negated")}
"""
            )

        contradiction_text = (
            "\n".join(contradiction_lines)
            if contradiction_lines
            else "None"
        )

        blocks.append(
            f"""
FIELD {i}

section_name:
{section_name}

parent_section:
{parent_section}

ORIGINAL CONTENT:
{original_content}

SUPPLIED FINDINGS FOR THIS FIELD:
{findings_text if findings_text else "None"}

CONTRADICTION CONTEXT FROM OTHER FIELDS:
{contradiction_text}

IMPORTANT:
- SUPPLIED FINDINGS may be incorporated into this field.
- CONTRADICTION CONTEXT must NOT be copied into this field.
- CONTRADICTION CONTEXT is provided ONLY so that you can remove
  an existing template statement that directly contradicts it.
- Do not add the contradiction-context finding itself unless it
  was also explicitly supplied for this field.

------------------------------------------------------------
"""
        )

    fields_block = "\n".join(blocks)

    return f"""
Edit the following template fields.

IMPORTANT:
- Return one edit for EVERY field.
- Preserve original wording whenever possible.
- Make only necessary changes.
- Incorporate all supplied findings for the field.
- Use contradiction context only to remove contradictory template text.
- Do not copy contradiction-context findings into the field.
- Do not invent information.
- Do not generate an impression.
- Do not modify fields not listed below.

{fields_block}

Return the final content for each field.
"""
# ============================================================
# BATCH EDITOR
# ============================================================
def remove_labrum_contradiction(
    edited_content,
    all_findings,
):
    """
    Remove contradictory normal labral statements only when there is
    a positive labral abnormality elsewhere in the extracted findings.
    """

    if not edited_content or not all_findings:
        return edited_content

    def get_finding_text(finding):
        if isinstance(finding, dict):
            return str(finding.get("finding") or "")

        return str(getattr(finding, "finding", "") or "")

    def is_negated(finding):
        if isinstance(finding, dict):
            return bool(finding.get("negated", False))

        return bool(getattr(finding, "negated", False))

    positive_labral_markers = (
        "labral fraying",
        "labral tear",
        "labral degeneration",
        "labral irregularity",
        "labral detachment",
        "labral lesion",
        "labral abnormality",
        "labrum fraying",
        "labrum tear",
        "labrum degeneration",
        "labrum irregularity",
        "labrum detachment",
        "labrum lesion",
        "labrum abnormality",
    )

    negative_labral_patterns = (
        "no labral tear",
        "no labral abnormality",
        "no labral lesion",
        "without labral tear",
        "without labral abnormality",
        "without labral lesion",
        "no tear of the labrum",
        "labrum is intact",
        "labrum intact",
    )

    has_positive_labral_abnormality = False

    for finding in all_findings:

        text = get_finding_text(finding).strip().lower()

        if not text:
            continue

        if is_negated(finding):
            continue

        if any(pattern in text for pattern in negative_labral_patterns):
            continue

        if any(marker in text for marker in positive_labral_markers):
            has_positive_labral_abnormality = True
            break

    if not has_positive_labral_abnormality:
        return edited_content

    patterns = [
        r"\bthe labrum is intact\.?\s*",
        r"\blabrum is intact\.?\s*",
        r"\bthe labrum appears intact\.?\s*",
        r"\blabrum appears intact\.?\s*",
    ]

    updated = edited_content

    for pattern in patterns:
        updated = re.sub(
            pattern,
            "",
            updated,
            flags=re.IGNORECASE,
        )

    return updated.strip()


def edit_fields_batch(
    fields: List[dict],
    all_findings=None
) -> List[dict]:

    if not fields:
        return []

    if all_findings is None:
        all_findings = []

    prompt = build_batch_field_edit_prompt(fields)

    response = generate_content_with_retry(

        client.models.generate_content,

        model=MODEL_NAME,

        contents=prompt,

        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=BatchFieldEditResult,
        ),

        fallback_prompt=prompt,

        fallback_system_instruction=SYSTEM_PROMPT,

        fallback_schema=BatchFieldEditResult,
    )

    # --------------------------------------------------------
    # Qwen / Gemini structured result
    # --------------------------------------------------------

    if isinstance(response, BatchFieldEditResult):

        result = response

    else:

        result = response.parsed

        if result is None:

            raise ValueError(
                "Gemini returned no structured batch field result."
            )

    # --------------------------------------------------------
    # Normalize model output
    # --------------------------------------------------------
    # Qwen may sometimes return the string "None" or "null"
    # instead of JSON null for root-level sections.

    for edit in result.edits:

        if isinstance(edit.parent_section, str):

            if edit.parent_section.strip().lower() in {
                "none",
                "null",
                "",
            }:

                edit.parent_section = None

        if isinstance(edit.section_name, str):

            edit.section_name = edit.section_name.strip()

    # --------------------------------------------------------
    # Deterministic labrum contradiction protection
    # --------------------------------------------------------
    #
    # If the model routes the labral abnormality correctly as an
    # insertion, the original GLENOHUMERAL JOINT field may still
    # contain:
    #
    #     The labrum is intact.
    #
    # Remove that contradiction deterministically.
    #
    # IMPORTANT:
    # If this cleanup changes the model output, force changed=True
    # so the cleanup is not overwritten by the "unchanged" logic.
    # --------------------------------------------------------

    for edit in result.edits:

        if (
            isinstance(edit.section_name, str)
            and edit.section_name.strip().upper()
            == "GLENOHUMERAL JOINT"
            and edit.parent_section
            and edit.parent_section.strip().upper()
            == "JOINTS"
        ):

            original_field = next(
                (
                    field
                    for field in fields
                    if (
                        field["section_name"] == edit.section_name
                        and field.get("parent_section")
                        == edit.parent_section
                    )
                ),
                None,
            )

            if original_field is None:
                continue

            cleaned_content = remove_labrum_contradiction(
                edit.edited_content,
                all_findings,
            )

            if cleaned_content != edit.edited_content.strip():

                edit.edited_content = cleaned_content

                # Deterministic cleanup itself constitutes a change.
                edit.changed = True

    # --------------------------------------------------------
    # Build deterministic input map
    # --------------------------------------------------------

    input_map = {
        (
            field["section_name"],
            field.get("parent_section")
        ): field
        for field in fields
    }

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    output = []

    for edit in result.edits:

        key = (
            edit.section_name,
            edit.parent_section
        )

        original_field = input_map.get(key)

        # ----------------------------------------------------
        # Ignore fields that were not requested.
        # ----------------------------------------------------

        if original_field is None:

            print(
                "[BatchEditor] Ignoring unexpected field:",
                edit.section_name
            )

            continue

        original_content = original_field["original_content"]

        # ----------------------------------------------------
        # If model says unchanged:
        #
        # ALWAYS preserve the exact original content.
        # ----------------------------------------------------

        if not edit.changed:

            edited_content = original_content

        else:

            edited_content = edit.edited_content.strip()

            # ------------------------------------------------
            # Safety: never allow an empty edit.
            # ------------------------------------------------

            if not edited_content:

                print(
                    "[BatchEditor] Empty edit detected. "
                    "Restoring original:",
                    edit.section_name
                )

                edited_content = original_content

                edit.changed = False

        output.append(
            {
                "section_name": edit.section_name,
                "parent_section": edit.parent_section,
                "changed": edit.changed,
                "original_content": original_content,
                "edited_content": edited_content,
            }
        )

    # --------------------------------------------------------
    # Safety:
    #
    # Make sure every requested field exists in the output.
    #
    # If Qwen/Gemini forgot a field, preserve the original
    # instead of allowing the field to disappear.
    # --------------------------------------------------------

    returned_keys = {
        (
            item["section_name"],
            item.get("parent_section")
        )
        for item in output
    }

    for field in fields:

        key = (
            field["section_name"],
            field.get("parent_section")
        )

        if key not in returned_keys:

            print(
                "[BatchEditor] Missing field from model output. "
                "Restoring original:",
                field["section_name"]
            )

            output.append(
                {
                    "section_name": field["section_name"],
                    "parent_section": field.get("parent_section"),
                    "changed": False,
                    "original_content": field["original_content"],
                    "edited_content": field["original_content"],
                }
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        f"[BatchEditor] Returning {len(output)} edited fields."
    )

    return output


  
# ============================================================
# BACKWARD-COMPATIBILITY WRAPPER
# ============================================================

def edit_fields(
    fields: List[dict],
    all_findings=None
) -> List[dict]:
    """
    Existing pipeline compatibility.

    Uses ONE batch LLM call instead of one call per field.
    """

    return edit_fields_batch(
        fields,
        all_findings=all_findings,
    )

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_fields = [

        {
            "section_name": "SUPRASPINATUS",
            "parent_section": "TENDONS",

            "original_content": (
                "No tear. The tendon is intact."
            ),

            "findings": [
                {
                    "finding": (
                        "Moderate supraspinatus tendinosis with "
                        "low-grade bursal-sided/intrasubstance "
                        "partial-thickness tear."
                    ),
                    "negated": False,
                    "laterality": None,
                    "measurement": None,
                    "action": "modify",
                }
            ],
        },

        {
            "section_name": "GLENOHUMERAL JOINT",
            "parent_section": "JOINTS",

            "original_content": (
                "No joint effusion. The joint spaces are maintained."
            ),

            "findings": [
                {
                    "finding": (
                        "Moderate glenohumeral joint effusion with "
                        "distention of the subscapularis recess."
                    ),
                    "negated": False,
                    "laterality": None,
                    "measurement": None,
                    "action": "modify",
                },
                {
                    "finding": (
                        "Mild diffuse chondral thinning without "
                        "advanced glenohumeral osteoarthritis."
                    ),
                    "negated": False,
                    "laterality": None,
                    "measurement": None,
                    "action": "modify",
                },
            ],
        },
    ]

    results = edit_fields_batch(test_fields)

    for result in results:

        print("\n==============================")
        print(result["section_name"])
        print("==============================")
        print(result["edited_content"])

