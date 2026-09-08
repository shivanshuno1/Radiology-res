#src/gemini_extractor.py
import os
import json
from typing import Optional, List, Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from template_structure import build_template_structure
from gemini_retry import generate_content_with_retry


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Create a .env file and add GEMINI_API_KEY=..."
    )

client = genai.Client(api_key=API_KEY)

# Use the model specified in .env.
#
# Example:
# GEMINI_MODEL=gemini-3.1-pro-preview
#
# If omitted, the default below is used.
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
)


# ============================================================
# PYDANTIC STRUCTURED OUTPUT SCHEMA
# ============================================================

class FindingChange(BaseModel):
    """
    One clinically meaningful patient-specific finding that requires
    either a template modification, insertion, or explicit no-change
    decision.
    """

    finding: str = Field(
        description=(
            "A clinically meaningful patient-specific finding explicitly "
            "supported by the dictation. Preserve important negation, "
            "severity, anatomy, laterality, measurements, and qualifiers. "
            "Do not add, infer, or invent information."
        )
    )

    negated: bool = Field(
        description=(
            "True only when the finding itself is explicitly negated in "
            "the dictation, such as 'no fracture' or 'without a tear'. "
            "False when the finding is present or abnormal. Do not treat "
            "a negated subcomponent as a positive finding."
        )
    )

    laterality: Optional[str] = Field(
        default=None,
        description=(
            "Explicit laterality associated with THIS SPECIFIC FINDING. "
            "Allowed values are 'right', 'left', 'bilateral', or null. "
            "NEVER copy laterality from the examination title, study "
            "description, body part, clinical history, or general context. "
            "For example, 'MRI RIGHT SHOULDER' does NOT make every "
            "finding right-sided. Use null unless right/left/bilateral "
            "is explicitly associated with the finding itself."
        )
    )

    measurement: Optional[str] = Field(
        default=None,
        description=(
            "The explicit measurement associated with the finding, "
            "preserved exactly as stated including value and unit, or "
            "null if no measurement is explicitly provided."
        )
    )

    target_section: Optional[str] = Field(
        default=None,
        description=(
            "The most specific EXISTING template section that contains "
            "the primary anatomical structure of the finding. The value "
            "must exactly match one of the supplied template section names. "
            "For 'insert', this must be null."
        )
    )

    parent_section: Optional[str] = Field(
        default=None,
        description=(
            "The exact existing parent section of target_section when "
            "target_section is a child section. Must exactly match the "
            "supplied template hierarchy. Use null when target_section "
            "is a root-level section or when action is 'insert'."
        )
    )

    action: Literal[
        "no_change",
        "modify",
        "insert"
    ] = Field(
        description=(
            "The required template action. "
            "'modify' = an existing template field must change. "
            "'insert' = the finding is clinically meaningful but there "
            "is no suitable existing template section. "
            "'no_change' = the template already correctly represents "
            "the finding and no edit is required."
        )
    )


class ExtractionResult(BaseModel):
    """
    Complete structured extraction from one dictation.
    """

    findings: List[FindingChange]


# ============================================================
# TEMPLATE SECTION EXTRACTION
# ============================================================

def get_template_sections(structure):
    """
    Flatten template hierarchy into a list of available sections.
    """

    sections = []

    def walk(nodes, parent=None):

        for node in nodes:

            name = node["name"]

            if name is not None:

                sections.append({
                    "name": name,
                    "parent": parent
                })

                walk(
                    node["children"],
                    name
                )

            else:

                walk(
                    node["children"],
                    parent
                )

    walk(structure["findings"])

    return sections


def format_template_sections(structure):
    """
    Format the actual template structure INCLUDING existing content.
    """

    lines = []

    def walk(sections, level=0):

        for section in sections:

            name = section.get("name")
            content = section.get("content", "")
            children = section.get("children", [])

            indent = "  " * level

            # ------------------------------------------------
            # Unlabelled content
            # ------------------------------------------------

            if name is None:

                if content:

                    lines.append(
                        f"{indent}[UNLABELLED]"
                    )

                    lines.append(
                        f"{indent}  {content}"
                    )

            # ------------------------------------------------
            # Named section
            # ------------------------------------------------

            else:

                lines.append(
                    f"{indent}{name}:"
                )

                if content:

                    for line in content.splitlines():

                        lines.append(
                            f"{indent}  {line}"
                        )

            # ------------------------------------------------
            # Children
            # ------------------------------------------------

            if children:

                walk(
                    children,
                    level + 1
                )

    walk(
        structure["findings"]
    )

    return "\n".join(lines)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a radiology report change-extraction and template-routing engine.

Your job is to identify the MINIMUM set of clinically meaningful,
patient-specific changes required to update an existing radiology report
template so that its FINDINGS accurately reflect the RADIOLOGY DICTATION.

You do NOT write the final report.

You do NOT write FINDINGS text.

You do NOT write an IMPRESSION.

You do NOT rewrite the complete template.

You do NOT explain your reasoning.

Return ONLY structured edit instructions matching the supplied
Pydantic schema.


============================================================
STRICT OUTPUT RULES
============================================================

Return ONE JSON OBJECT.

Never return a top-level JSON array.

If the schema contains "findings", the structure MUST be:

{
  "findings": [
    {
      "finding": "...",
      "negated": false,
      "laterality": null,
      "measurement": null,
      "target_section": "...",
      "parent_section": "...",
      "action": "modify"
    }
  ]
}

Return ONLY valid JSON.

Do not use Markdown.

Do not use code fences.

Do not write explanations.

Do not write reasoning.

Do not repeat the dictation.

Do not repeat the template.

Do not generate an impression.

Do not generate the complete report.


============================================================
1. SOURCE OF TRUTH
============================================================

The RADIOLOGY DICTATION is the ONLY source of patient-specific
findings.

The template is NOT evidence that a finding exists.

The template impression is NOT evidence that a finding exists.

Use only information explicitly supported by the dictation.

Never invent findings.

Never infer findings from:

- modality
- body part
- study description
- patient age
- patient sex
- clinical history
- indication
- expected anatomy
- typical radiology patterns
- template wording
- template impression


============================================================
2. CORE OBJECTIVE
============================================================

Compare:

ORIGINAL TEMPLATE
+
RADIOLOGY DICTATION

and identify only the changes actually required.

Return the SMALLEST ACCURATE SET OF EDITS.

Do not extract every sentence.

Do not extract technique.

Do not extract history.

Do not extract indication.

Do not extract routine normal findings.

Do not extract unchanged information.


============================================================
3. TEMPLATE-AWARE EDIT DECISION
============================================================

Always inspect the actual content of the relevant template section.

A finding requires an edit only when:

1. The dictation contradicts the template.

OR

2. The dictation adds clinically meaningful information not represented
   by the template.

OR

3. A clinically meaningful finding has no suitable template field.

If the template already correctly represents the dictated information,
omit the finding.

Do not modify wording unnecessarily.

Do not modify unmentioned sections.


============================================================
4. ACTION TYPES
============================================================

Use:

"modify"

when an existing template section must change.

Use:

"insert"

only when the finding is clinically meaningful AND there is no
appropriate existing specific or broader section.

Use:

"no_change"

only when genuinely necessary.

Normally omit unchanged findings.


============================================================
5. ANATOMICAL ROUTING
============================================================

Route each finding according to its PRIMARY anatomical structure.

Use:

exact existing section
        ↓
if unavailable:
appropriate broader anatomical section
        ↓
if unavailable:
INSERT


============================================================
6. DO NOT CROSS-ROUTE
============================================================

Anatomical association does NOT equal ownership.

Examples:

supraspinatus tear
→ SUPRASPINATUS

subscapularis tendinosis
→ SUBSCAPULARIS

long head biceps tenosynovitis
→ BICEPS BRACHII, LONG HEAD if present

glenohumeral effusion
→ GLENOHUMERAL JOINT

AC joint osteoarthritis
→ ACROMIOCLAVICULAR JOINT

greater tuberosity marrow abnormality
→ GREATER TUBEROSITY if present
→ otherwise BONES

labral pathology
→ LABRUM if present
→ otherwise INSERT

subacromial-subdeltoid bursitis
→ BURSA/BURSAL section if present
→ otherwise INSERT

rotator interval abnormality
→ ROTATOR INTERVAL if present
→ otherwise CAPSULE if appropriate
→ otherwise INSERT


============================================================
7. CRITICAL SHOULDER ROUTING OVERRIDES
============================================================

LABRUM:

If LABRUM exists:
→ LABRUM

If LABRUM does not exist:
→ INSERT

NEVER route labral pathology to GLENOHUMERAL JOINT merely because the
labrum is associated with that joint.


BURSA:

If BURSA or BURSAL exists:
→ that section

Otherwise:
→ INSERT

NEVER route bursitis to a joint section merely because the bursa is
adjacent to a joint.


LONG HEAD OF BICEPS:

If BICEPS BRACHII, LONG HEAD exists:
→ BICEPS BRACHII, LONG HEAD

Do NOT INSERT when that section exists.


ROTATOR INTERVAL / CAPSULE:

Use a specific existing section if available.

Otherwise:
→ INSERT

Do not automatically route these to GLENOHUMERAL JOINT.


ACROMION:

If ACROMION exists:
→ ACROMION

Otherwise:
→ BONES if appropriate.


GREATER TUBEROSITY:

If GREATER TUBEROSITY exists:
→ GREATER TUBEROSITY

Otherwise:
→ BONES if appropriate.


============================================================
8. SECTION HIERARCHY
============================================================

For child sections, target_section must be the child.

Example:

TENDONS
  SUPRASPINATUS

returns:

target_section = "SUPRASPINATUS"
parent_section = "TENDONS"

Never use the parent when a specific child exists.

Root-level section:

parent_section = null

INSERT:

target_section = null
parent_section = null


============================================================
9. MULTIPLE FINDINGS IN ONE SECTION
============================================================

Multiple independent findings may target the same section.

Example:

Template:

GLENOHUMERAL JOINT:
No joint effusion. Cartilage is preserved.

Dictation:

Moderate glenohumeral joint effusion.
Mild diffuse chondral thinning.

These are two independent findings.

Return both.

Do not discard one because they share a target section.


============================================================
10. GROUPING
============================================================

Combine closely related findings involving the SAME structure and SAME
pathological process.

Example:

"Moderate supraspinatus tendinosis with low-grade partial-thickness
tear"

is ONE finding.

Keep unrelated structures separate.


============================================================
11. DUPLICATE CONSOLIDATION
============================================================

The dictation may describe the same finding multiple times.

Do not create duplicate edits.

If a later summary adds specificity, retain the most specific information
explicitly supported by the dictation.


============================================================
12. NEGATION
============================================================

Preserve negation.

Examples:

"No fracture"

→ finding = "fracture"
→ negated = true

"No pleural effusion"

→ finding = "pleural effusion"
→ negated = true

"Small pleural effusion"

→ negated = false


============================================================
13. NEGATED SUBCOMPONENTS
============================================================

If a finding contains both positive and negative information, the overall
finding remains positive when the primary pathology is present.

Example:

"Mild insertional subscapularis tendinosis without a discrete tear"

means:

tendinosis = present
discrete tear = absent

Therefore:

negated = false

Keep the negative qualifier in finding text.


============================================================
14. LATERALITY — EXTREMELY IMPORTANT
============================================================

LATERALITY IS A FINDING-LEVEL ATTRIBUTE.

Allowed values:

"right"
"left"
"bilateral"
null


HARD RULE:

NEVER copy laterality from the examination title.

NEVER copy laterality from the study description.

NEVER copy laterality from the body part.

NEVER copy laterality from clinical history.

NEVER propagate laterality from one finding to another.

NEVER assume all findings share the side mentioned in the examination
title.


For example:

"MRI RIGHT SHOULDER WITHOUT CONTRAST"

does NOT mean:

supraspinatus → right
biceps → right
labrum → right
bursitis → right

unless each finding itself explicitly establishes that laterality.


CORRECT:

"Mild right pleural effusion"

→ laterality = "right"


CORRECT:

"Left supraspinatus tear"

→ laterality = "left"


CORRECT:

"Bilateral knee effusions"

→ laterality = "bilateral"


CORRECT:

"Supraspinatus tendinosis"

→ laterality = null


CORRECT:

"Moderate glenohumeral joint effusion"

→ laterality = null


For the phrase:

"MRI RIGHT SHOULDER"

the word RIGHT belongs to the examination identity,
NOT automatically to individual findings.

When there is ANY doubt:

laterality = null


============================================================
15. MEASUREMENTS
============================================================

Preserve explicit measurements exactly.

Examples:

6 mm cyst
→ measurement = "6 mm"

2.3 cm lesion
→ measurement = "2.3 cm"

12 x 8 mm lesion
→ measurement = "12 x 8 mm"

Do not invent or estimate measurements.


============================================================
16. SEVERITY AND QUALIFIERS
============================================================

Preserve explicitly stated qualifiers such as:

mild
moderate
severe
minimal
trace
small
large
acute
chronic
low-grade
high-grade
partial-thickness
full-thickness
displaced
nondisplaced
focal
diffuse
proximal
distal
insertional
degenerative
reactive

Do not upgrade or downgrade severity.


============================================================
17. DIAGNOSIS VS DESCRIPTION
============================================================

Do not convert uncertain descriptions into definitive diagnoses.

Example:

"mild thickening and edema within the rotator interval, which may be
seen with adhesive capsulitis"

must remain uncertain.

Do not change it automatically to:

"adhesive capsulitis"


============================================================
18. NORMAL FINDINGS
============================================================

Routine normal findings already represented correctly by the template
should normally be omitted.

Example:

Template:
INFRASPINATUS:
The tendon is intact.

Dictation:
Infraspinatus tendon is intact.

→ OMIT


============================================================
19. PRESERVE TEMPLATE CONTENT
============================================================

Do not remove template information simply because the dictation does not
mention it.

The downstream editor must preserve correct unmentioned content.

Only identify the section requiring change.


============================================================
20. FINDING FIELD
============================================================

The finding must be:

- concise
- clinically meaningful
- explicitly supported
- anatomically specific

Preserve:

- pathology
- negation
- severity
- explicitly associated laterality
- measurements
- qualifiers
- uncertainty


============================================================
21. FINAL VALIDATION
============================================================

Before returning JSON, verify internally:

1. Every finding comes from the dictation.
2. No finding came from the template alone.
3. No diagnosis was invented.
4. Negation is correct.
5. Laterality is explicitly associated with the finding.
6. Study-level laterality was NOT propagated.
7. Measurements are preserved.
8. Severity is preserved.
9. Correct anatomical owner was selected.
10. Specific section was checked first.
11. Broader section was considered second.
12. INSERT is used only when necessary.
13. Labrum is not cross-routed to GLENOHUMERAL JOINT.
14. Bursitis is not cross-routed to a joint.
15. Biceps tenosynovitis uses BICEPS BRACHII, LONG HEAD when present.
16. Every MODIFY target exists.
17. Every parent matches the hierarchy.
18. Duplicate findings are consolidated.
19. Routine unchanged findings are omitted.
20. No impression is generated.
21. No complete report is generated.
22. No explanation is generated.
23. Output is one complete valid JSON object.


============================================================
22. OUTPUT
============================================================

Return ONLY the JSON object required by the Pydantic schema.

No reasoning.

No explanation.

No Markdown.

No report.

No impression.

No commentary.
"""


# ============================================================
# EXTRACTION PROMPT
# ============================================================

def build_extraction_prompt(dictation, structure):

    sections_text = format_template_sections(
        structure
    )

    prompt = f"""
{SYSTEM_PROMPT}


============================================================
AVAILABLE TEMPLATE SECTIONS
============================================================

{sections_text}


============================================================
RADIOLOGY DICTATION
============================================================

{dictation}


============================================================
TASK
============================================================

Identify the minimum set of clinically meaningful changes required
to update the template.

For every returned finding:

- use only information explicitly supported by the dictation
- preserve clinically meaningful qualifiers
- preserve explicit negation
- preserve explicit measurements
- use laterality ONLY when explicitly associated with that finding
- DO NOT inherit laterality from the examination title
- select the most specific existing template section
- use the correct parent section
- use INSERT only when no appropriate existing section exists
- omit routine unchanged normal findings
- consolidate duplicate descriptions

Remember:

The examination title is NOT evidence of finding-level laterality.

The task is CHANGE EXTRACTION.

Do NOT write the final report.

Do NOT generate FINDINGS.

Do NOT generate IMPRESSION.
"""

    return prompt


# ============================================================
# GEMINI / QWEN EXTRACTION
# ============================================================

def extract_findings(dictation, structure):

    prompt = build_extraction_prompt(
        dictation=dictation,
        structure=structure
    )

    response = generate_content_with_retry(
        client.models.generate_content,

        model=MODEL_NAME,

        contents=prompt,

        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ExtractionResult,
        ),

        fallback_prompt=prompt,

        fallback_system_instruction=SYSTEM_PROMPT,

        fallback_schema=ExtractionResult,
    )

    # --------------------------------------------------------
    # Get structured result
    # --------------------------------------------------------

    if isinstance(response, ExtractionResult):

        result = response

    else:

        result = response.parsed

        if result is None:
            raise ValueError(
                "Gemini returned no structured response."
            )

    # --------------------------------------------------------
    # IMPORTANT:
    # Apply deterministic routing AFTER LLM extraction.
    # --------------------------------------------------------

    result = apply_hard_routing_rules(
        result=result,
        structure=structure
    )

    result = apply_labrum_hard_rule(
        result=result,
        dictation=dictation,
        structure=structure
    )

    # --------------------------------------------------------
    # Validate AFTER hard routing.
    # --------------------------------------------------------

    validate_result(
        result=result,
        structure=structure
    )

    return result


def apply_hard_routing_rules(result, structure):
    """
    Deterministic post-processing after LLM extraction.

    The LLM identifies the clinical finding.
    This function enforces high-confidence anatomical routing,
    section hierarchy, and metadata safety.

    IMPORTANT:
    This function must NOT delete valid findings simply because
    they were not routed by a hard-coded rule.

    Routing priority is intentional:
        1. Bone-specific findings
        2. Labrum
        3. True bursal pathology
        4. Glenohumeral joint
        5. Acromioclavicular joint
        6. Rotator interval / capsule
        7. Individual tendons
        8. Acromion

    This ordering prevents substring-based false routing such as:

        "bursal-sided supraspinatus tear"
            -> incorrectly routed to BURSA

    and:

        "glenohumeral effusion with distention of the
         subscapularis recess"
            -> incorrectly routed to SUBSCAPULARIS tendon.
    """

    # ========================================================
    # AVAILABLE TEMPLATE SECTIONS
    # ========================================================

    sections = get_template_sections(structure)

    section_map = {
        section["name"].upper(): section
        for section in sections
        if section.get("name")
    }

    def normalize_optional_section(value):
        """
        Convert LLM strings such as 'None', 'null', or 'N/A'
        into actual Python None.
        """

        if value is None:
            return None

        value = str(value).strip()

        if value.lower() in {
            "",
            "none",
            "null",
            "n/a",
            "na",
        }:
            return None

        return value

    def get_section(name):
        """
        Return the actual template section using
        case-insensitive lookup.
        """

        if not name:
            return None

        return section_map.get(
            str(name).strip().upper()
        )

    def find_bursal_section():
        """
        Find an existing BURSA/BURSAE/BURSAL section.

        Returns the actual section or None.
        """

        for section in sections:

            name = str(
                section.get("name", "")
            ).strip().upper()

            if (
                "BURSA" in name
                or "BURSAL" in name
            ):
                return section

        return None

    def find_rotator_interval_section():
        """
        Find a dedicated rotator interval/capsule section,
        if the template provides one.
        """

        for section in sections:

            name = str(
                section.get("name", "")
            ).strip().upper()

            if (
                "ROTATOR INTERVAL" in name
                or name == "CAPSULE"
            ):
                return section

        return None

    # ========================================================
    # CLEAN FINDINGS
    # ========================================================

    cleaned = []

    for finding in result.findings:

        # ----------------------------------------------------
        # Local closures that operate on the current `finding`
        # ----------------------------------------------------

        def route_to(name, finding=finding):
            """
            Route the current finding to an existing template
            section. Returns True if the section exists.
            """

            section = get_section(name)

            if section is None:
                return False

            finding.target_section = section["name"]

            finding.parent_section = normalize_optional_section(
                section.get("parent")
            )

            finding.action = "modify"

            return True

        def route_insert(finding=finding):
            """
            Mark the current finding as an insertion.

            Insertions intentionally have no target or parent
            because the template does not contain a suitable
            labelled section.
            """

            finding.target_section = None
            finding.parent_section = None
            finding.action = "insert"

        # ----------------------------------------------------
        # Reset routing before applying hard rules
        # ----------------------------------------------------

        original_target = finding.target_section
        original_parent = finding.parent_section

        # ----------------------------------------------------
        # Normalize text
        # ----------------------------------------------------

        text = (
            finding.finding
            if finding.finding is not None
            else ""
        )

        text = str(text).strip()

        finding.finding = text

        text_lower = text.lower()

        # ----------------------------------------------------
        # Normalize LLM section values
        # ----------------------------------------------------

        finding.target_section = normalize_optional_section(
            finding.target_section
        )

        finding.parent_section = normalize_optional_section(
            finding.parent_section
        )

        # ====================================================
        # SHOULDER-SPECIFIC ROUTING
        # ====================================================

        # ----------------------------------------------------
        # GREATER TUBEROSITY / BONE FINDINGS
        #
        # MUST COME BEFORE SUPRASPINATUS.
        #
        # Example:
        #
        # "cystic changes in the greater tuberosity at the
        # supraspinatus insertion"
        #
        # The pathology is osseous, even though
        # "supraspinatus" appears in the sentence.
        # ----------------------------------------------------

        if (
            "greater tuberosity" in text_lower
            or "subcortical cyst" in text_lower
            or "reactive marrow edema" in text_lower
            or (
                "marrow edema" in text_lower
                and (
                    "bone" in text_lower
                    or "osseous" in text_lower
                )
            )
        ):

            if route_to("GREATER TUBEROSITY"):

                print(
                    "[HardRouting] GREATER TUBEROSITY -> "
                    "GREATER TUBEROSITY"
                )

            elif route_to("BONES"):

                print(
                    "[HardRouting] GREATER TUBEROSITY -> "
                    "BONES"
                )

            else:

                print(
                    "[HardRouting] No bone section found; "
                    "marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # LABRUM / LABRAL
        #
        # Labral pathology must NOT be routed to the
        # GLENOHUMERAL JOINT simply because the labrum
        # is anatomically related to that joint.
        #
        # If the template has no LABRUM field, preserve
        # the finding as an insertion.
        # ----------------------------------------------------

        elif (
            "labrum" in text_lower
            or "labral" in text_lower
        ):

            if route_to("LABRUM"):

                print(
                    "[HardRouting] LABRUM -> LABRUM"
                )

            else:

                print(
                    "[Postprocess] LABRUM section not present; "
                    "marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # TRUE BURSA / BURSITIS
        #
        # IMPORTANT:
        #
        # NEVER use:
        #
        #     "bursa" in text_lower
        #
        # because:
        #
        #     "bursal-sided"
        #
        # would incorrectly be interpreted as a bursal
        # abnormality.
        #
        # We therefore look for actual bursal pathology.
        # ----------------------------------------------------

        elif (
            "bursitis" in text_lower
            or "subacromial-subdeltoid bursa" in text_lower
            or "subacromial bursa" in text_lower
            or "subdeltoid bursa" in text_lower
            or "bursal fluid" in text_lower
            or "fluid in the bursa" in text_lower
        ):

            bursal_section = find_bursal_section()

            if bursal_section is not None:

                finding.target_section = bursal_section["name"]

                finding.parent_section = normalize_optional_section(
                    bursal_section.get("parent")
                )

                finding.action = "modify"

                print(
                    "[HardRouting] BURSA/BURSITIS ->",
                    bursal_section["name"]
                )

            else:

                print(
                    "[HardRouting] No bursa section; "
                    "marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # GLENOHUMERAL JOINT
        #
        # THIS MUST COME BEFORE SUBSCAPULARIS.
        #
        # Example:
        #
        # "moderate glenohumeral joint effusion with
        # distention of the subscapularis recess"
        #
        # is a GLENOHUMERAL JOINT finding.
        #
        # It must NOT become:
        #
        # TENDONS -> SUBSCAPULARIS
        #
        # merely because "subscapularis" appears in the
        # description.
        # ----------------------------------------------------

        elif (
            "glenohumeral joint" in text_lower
            or "glenohumeral effusion" in text_lower
            or "glenohumeral osteoarthritis" in text_lower
            or "glenohumeral osteoarthrosis" in text_lower
            or "glenohumeral arthrosis" in text_lower
            or "glenohumeral chondral" in text_lower
        ):

            if route_to("GLENOHUMERAL JOINT"):

                print(
                    "[HardRouting] GLENOHUMERAL -> "
                    "JOINTS -> GLENOHUMERAL JOINT"
                )

            else:

                print(
                    "[HardRouting] No glenohumeral joint "
                    "section; marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # ACROMIOCLAVICULAR JOINT
        # ----------------------------------------------------

        elif (
            "acromioclavicular joint" in text_lower
            or "acromioclavicular osteoarthritis" in text_lower
            or "acromioclavicular osteoarthrosis" in text_lower
            or "acromioclavicular arthrosis" in text_lower
        ):

            if route_to("ACROMIOCLAVICULAR JOINT"):

                print(
                    "[HardRouting] ACROMIOCLAVICULAR -> "
                    "JOINTS -> ACROMIOCLAVICULAR JOINT"
                )

            else:

                print(
                    "[HardRouting] No AC joint section; "
                    "marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # ROTATOR INTERVAL / CAPSULE
        #
        # Check specific phrases rather than simply:
        #
        #     "capsule" in text_lower
        #
        # because "capsular hypertrophy" can occur as part
        # of AC joint pathology and should remain with the
        # AC joint finding.
        # ----------------------------------------------------

        elif (
            "rotator interval" in text_lower
            or "inferior glenohumeral capsule" in text_lower
            or "inferior capsular thickening" in text_lower
            or "inferior capsular edema" in text_lower
            or "capsular thickening" in text_lower
            or (
                "capsular edema" in text_lower
                and "acromioclavicular" not in text_lower
            )
        ):

            specific_section = find_rotator_interval_section()

            if specific_section is not None:

                finding.target_section = specific_section["name"]

                finding.parent_section = normalize_optional_section(
                    specific_section.get("parent")
                )

                finding.action = "modify"

                print(
                    "[HardRouting] "
                    "ROTATOR INTERVAL/CAPSULE ->",
                    specific_section["name"]
                )

            else:

                print(
                    "[HardRouting] No specific "
                    "rotator interval/capsule section; "
                    "marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # SUPRASPINATUS
        #
        # This comes AFTER:
        #
        # - greater tuberosity
        # - labrum
        # - true bursa
        # - glenohumeral joint
        # - AC joint
        #
        # This prevents "bursal-sided supraspinatus tear"
        # from being routed to BURSA.
        # ----------------------------------------------------

        elif "supraspinatus" in text_lower:

            if route_to("SUPRASPINATUS"):

                print(
                    "[HardRouting] SUPRASPINATUS -> "
                    "TENDONS -> SUPRASPINATUS"
                )

            else:

                print(
                    "[HardRouting] SUPRASPINATUS section "
                    "not found; marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # INFRASPINATUS
        # ----------------------------------------------------

        elif "infraspinatus" in text_lower:

            if route_to("INFRASPINATUS"):

                print(
                    "[HardRouting] INFRASPINATUS -> "
                    "TENDONS -> INFRASPINATUS"
                )

            else:

                print(
                    "[HardRouting] INFRASPINATUS section "
                    "not found; marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # TERES MINOR
        # ----------------------------------------------------

        elif "teres minor" in text_lower:

            if route_to("TERES MINOR"):

                print(
                    "[HardRouting] TERES MINOR -> "
                    "TENDONS -> TERES MINOR"
                )

            else:

                print(
                    "[HardRouting] TERES MINOR section "
                    "not found; marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # LONG HEAD OF BICEPS
        #
        # Check long-head biceps after the other high-confidence
        # shoulder structures.
        # ----------------------------------------------------

        elif (
            "biceps" in text_lower
            and (
                "long head" in text_lower
                or "biceps tendon" in text_lower
                or "biceps tendon sheath" in text_lower
                or "biceps tenosynovitis" in text_lower
            )
        ):

            if route_to("BICEPS BRACHII, LONG HEAD"):

                print(
                    "[HardRouting] BICEPS -> "
                    "TENDONS -> BICEPS BRACHII, LONG HEAD"
                )

            elif route_to("BICEPS BRACHII"):

                print(
                    "[HardRouting] BICEPS -> "
                    "TENDONS -> BICEPS BRACHII"
                )

            else:

                print(
                    "[HardRouting] No biceps section; "
                    "marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # SUBSCAPULARIS
        #
        # IMPORTANT:
        #
        # This comes AFTER GLENOHUMERAL JOINT.
        #
        # Therefore:
        #
        # "glenohumeral effusion with distention of the
        # subscapularis recess"
        #
        # is correctly handled by the joint rule above.
        #
        # A true tendon finding such as:
        #
        # "subscapularis tendinosis"
        #
        # reaches this block.
        # ----------------------------------------------------

        elif "subscapularis" in text_lower:

            if route_to("SUBSCAPULARIS"):

                print(
                    "[HardRouting] SUBSCAPULARIS -> "
                    "TENDONS -> SUBSCAPULARIS"
                )

            else:

                print(
                    "[HardRouting] SUBSCAPULARIS section "
                    "not found; marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ----------------------------------------------------
        # ACROMION
        #
        # This is checked after AC joint because an AC joint
        # finding can mention the acromion.
        # ----------------------------------------------------

        elif "acromion" in text_lower:

            if route_to("ACROMION"):

                print(
                    "[HardRouting] ACROMION -> ACROMION"
                )

            elif route_to("BONES"):

                print(
                    "[HardRouting] ACROMION -> BONES"
                )

            else:

                print(
                    "[HardRouting] No acromion/bone section; "
                    "marking as insertion:",
                    finding.finding
                )

                route_insert()

        # ====================================================
        # LATERALITY SAFETY
        # ====================================================

        if finding.laterality is not None:

            laterality = str(finding.laterality).strip().lower()

            if laterality not in {
                "right",
                "left",
                "bilateral",
            }:

                print(
                    "[Postprocess] Invalid laterality:",
                    finding.laterality
                )

                finding.laterality = None

            elif laterality not in text_lower:

                print(
                    "[Postprocess] Removing unsupported "
                    "laterality:",
                    finding.laterality,
                    "from:",
                    finding.finding
                )

                finding.laterality = None

            else:

                finding.laterality = laterality

        # ====================================================
        # FINAL ROUTING SANITY
        # ====================================================

        # If this is a MODIFY action, make sure the target
        # actually exists and the parent is correct.

        if finding.action == "modify":

            target = finding.target_section

            if target is None:

                print(
                    "[Postprocess] MODIFY without target; "
                    "converting to INSERT:",
                    finding.finding
                )

                route_insert()

            else:

                actual_section = get_section(target)

                if actual_section is None:

                    print(
                        "[Postprocess] Invalid target section:",
                        target,
                        "-> converting to INSERT"
                    )

                    route_insert()

                else:

                    # Always use the actual template name.
                    finding.target_section = actual_section["name"]

                    # Always derive parent from the actual
                    # template structure rather than trusting
                    # the LLM.
                    finding.parent_section = normalize_optional_section(
                        actual_section.get("parent")
                    )

        # ====================================================
        # FINAL INSERT SANITY
        # ====================================================

        if finding.action == "insert":

            finding.target_section = None
            finding.parent_section = None

        # ====================================================
        # ADD TO CLEANED RESULT
        # ====================================================

        cleaned.append(finding)

    # ========================================================
    # REPLACE ORIGINAL FINDINGS
    # ========================================================

    result.findings = cleaned

    return result


# ============================================================
# LABRUM RULE
# ============================================================

def apply_labrum_hard_rule(result, dictation, structure):
    """
    Deterministically preserve explicit positive labral abnormalities.

    Purpose:
    - Recover a labral abnormality if Gemini completely missed it.
    - NEVER invent clinical wording.
    - NEVER hardcode severity, anatomy, tear status, or qualifiers.
    - Recover the actual finding from the supplied dictation.
    - Insert the finding when the template has no LABRUM section.
    """

    import re

    dictation_text = str(dictation or "").strip()

    if not dictation_text:
        return result

    dictation_lower = dictation_text.lower()

    # ========================================================
    # 1. Detect explicit positive labral pathology
    # ========================================================

    positive_labral_patterns = (
        "labral fraying",
        "labral degeneration",
        "labral tear",
        "labral lesion",
        "labral irregularity",
        "labral abnormality",
        "labrum shows",
        "labrum demonstrates",
        "labrum has",
        "degenerative fraying",
    )

    if not any(
        pattern in dictation_lower
        for pattern in positive_labral_patterns
    ):
        return result

    # ========================================================
    # 2. Check whether Gemini already extracted it
    # ========================================================

    for finding in result.findings:

        text = str(finding.finding or "").lower()

        if (
            ("labrum" in text or "labral" in text)
            and not finding.negated
        ):
            return result

    # ========================================================
    # 3. Recover the ACTUAL sentence from dictation
    # ========================================================

    sentences = re.split(
        r"(?<=[.!?])\s+",
        dictation_text,
    )

    abnormality_markers = (
        "fraying",
        "degeneration",
        "degenerative",
        "tear",
        "lesion",
        "irregularity",
        "abnormal",
        "detachment",
        "fibrillation",
        "defect",
    )

    normal_labral_phrases = (
        "no labral abnormality",
        "no labral tear",
        "no labral lesion",
        "labrum is intact",
        "labrum appears intact",
        "labrum is normal",
        "normal labrum",
    )

    labrum_sentence = None

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_lower = sentence.lower()

        # Must explicitly refer to labrum/labral anatomy.
        if (
            "labrum" not in sentence_lower
            and "labral" not in sentence_lower
        ):
            continue

        # Do not recover a purely normal/negative statement.
        if any(
            phrase in sentence_lower
            for phrase in normal_labral_phrases
        ):
            continue

        # Must contain an explicit abnormality.
        if not any(
            marker in sentence_lower
            for marker in abnormality_markers
        ):
            continue

        labrum_sentence = sentence
        break

    # ========================================================
    # 4. Safety: NEVER synthesize a finding
    # ========================================================

    if labrum_sentence is None:

        print(
            "[HardRouting] Positive labral pathology detected, "
            "but exact dictation wording could not be recovered. "
            "Skipping synthetic labrum finding."
        )

        return result

    # ========================================================
    # 5. Add the actual dictation finding
    #
    # No LABRUM section is forced.
    # The downstream template editor decides insertion.
    # ========================================================

    result.findings.append(
        FindingChange(
            finding=labrum_sentence,
            negated=False,
            laterality=None,
            measurement=None,
            target_section=None,
            parent_section=None,
            action="insert",
        )
    )

    print(
        "[HardRouting] Recovered explicit LABRUM abnormality "
        "from dictation -> INSERT"
    )

    return result



def is_redundant_negative(finding, section_content):
    """
    Returns True when a negated finding is already explicitly represented
    by the template section and therefore should not trigger an edit.
    """

    if not finding.negated:
        return False

    finding_text = finding.finding.lower()
    section_text = section_content.lower()

    # Full-thickness rotator cuff tear
    if "full-thickness rotator cuff tear" in finding_text:
        if "no full-thickness rotator cuff tear" in section_text:
            return True

    # Fracture
    if "acute fracture" in finding_text:
        if "no acute fracture" in section_text:
            return True

    # Aggressive osseous lesion
    if "aggressive osseous lesion" in finding_text:
        if "no aggressive osseous lesion" in section_text:
            return True

    return False


def validate_result(result, structure):

    validated = []

    for finding in result.findings:

        # ------------------------------------------------
        # Reject completely empty findings
        # ------------------------------------------------

        if (
            not finding.finding
            or not finding.finding.strip()
        ):

            print(
                "[Validation] Empty finding detected. "
                "Ignoring."
            )

            continue

        # ------------------------------------------------
        # Normalize empty section values
        # ------------------------------------------------

        if (
            isinstance(
                finding.target_section,
                str
            )
            and not finding.target_section.strip()
        ):

            finding.target_section = None

        if (
            isinstance(
                finding.parent_section,
                str
            )
            and not finding.parent_section.strip()
        ):

            finding.parent_section = None

        # ------------------------------------------------
        # INSERT findings must not retain a target.
        # ------------------------------------------------

        if finding.action == "insert":

            finding.target_section = None
            finding.parent_section = None

        # ------------------------------------------------
        # NO_CHANGE findings must not modify a section.
        # ------------------------------------------------

        elif finding.action == "no_change":

            finding.target_section = None
            finding.parent_section = None

        # ------------------------------------------------
        # Preserve all valid findings.
        #
        # Detailed routing has already been handled by:
        #
        # apply_hard_routing_rules()
        # apply_labrum_hard_rule()
        # ------------------------------------------------

        validated.append(finding)

    result.findings = validated

    return result
# ============================================================
# PRINT RESULT
# ============================================================

def print_extraction(result):

    print("\n")
    print("=" * 80)
    print("STRUCTURED EXTRACTION")
    print("=" * 80)

    if not result.findings:

        print("No findings extracted.")

        return

    for i, finding in enumerate(result.findings, start=1):

        print()
        print(f"FINDING {i}")
        print("-" * 40)
        print("Finding:", finding.finding)
        print("Negated:", finding.negated)
        print("Laterality:", finding.laterality)
        print("Measurement:", finding.measurement)
        print("Target:", finding.target_section)
        print("Parent:", finding.parent_section)
        print("Action:", finding.action)


# ============================================================
# SAVE JSON
# ============================================================

def save_result(result, filename):

    data = result.model_dump()

    with open(filename, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    import json
    import traceback

    print("=" * 100)
    print("RADIOLOGY EXTRACTION DEBUG")
    print("=" * 100)

    # ========================================================
    # 1. LOAD TRAIN DATA
    # ========================================================

    try:
        train = pd.read_csv("data/train.csv")

        print(f"\n[1/7] TRAIN DATA LOADED")
        print(f"      Number of cases : {len(train)}")
        print(f"      Columns         : {list(train.columns)}")

    except Exception as e:
        print("\n[ERROR] Could not load train.csv")
        print(f"Reason: {e}")
        traceback.print_exc()
        raise

    # ========================================================
    # 2. SELECT SHOULDER EXAMPLE
    # ========================================================

    print("\n" + "=" * 100)
    print("[2/7] SELECTING SHOULDER EXAMPLE")
    print("=" * 100)

    mask = (
        train["dictation"]
        .fillna("")
        .str.contains("supraspinatus", case=False, na=False)
        &
        train["dictation"]
        .fillna("")
        .str.contains("glenohumeral", case=False, na=False)
    )

    matching_cases = train[mask]

    print(f"Matching cases found: {len(matching_cases)}")

    if matching_cases.empty:
        raise RuntimeError(
            "No shoulder case found containing both "
            "'supraspinatus' and 'glenohumeral'."
        )

    selected = matching_cases.iloc[0]

    print("\nSELECTED CASE")
    print("-" * 100)
    print("Case ID:")
    print(selected["case_id"])

    print("\nModality:")
    print(selected["modality"])

    print("\nBody Part:")
    print(selected["body_part"])

    print("\nStudy Description:")
    print(selected["study_description"])

    print("\nPatient Age Band:")
    print(selected["patient_age_band"])

    print("\nPatient Sex:")
    print(selected["patient_sex"])

    # ========================================================
    # 3. PRINT DICTATION
    # ========================================================

    print("\n" + "=" * 100)
    print("[3/7] DICTATION")
    print("=" * 100)

    dictation = str(selected["dictation"]).strip()

    print(dictation)

    # ========================================================
    # 4. BUILD TEMPLATE STRUCTURE
    # ========================================================

    print("\n" + "=" * 100)
    print("[4/7] BUILDING TEMPLATE STRUCTURE")
    print("=" * 100)

    template = str(selected["template_content"])

    structure = build_template_structure(template)

    print("\nRAW TEMPLATE")
    print("-" * 100)
    print(template)

    print("\nPARSED TEMPLATE STRUCTURE")
    print("-" * 100)

    def print_structure(sections, level=0):
        indent = "    " * level

        for section in sections:

            name = section.get("name")
            content = section.get("content", "")
            children = section.get("children", [])

            if name is None:
                print(f"{indent}[UNLABELLED]")
            else:
                print(f"{indent}[SECTION] {name}")

            if content:
                print(f"{indent}  Content: {content}")

            if children:
                print_structure(children, level + 1)

    print_structure(structure["findings"])

    print("\nTEMPLATE IMPRESSION")
    print("-" * 100)
    print(structure["impression"])

    # ========================================================
    # 5. GEMINI / QWEN EXTRACTION
    # ========================================================

    print("\n" + "=" * 100)
    print("[5/7] EXTRACTING FINDINGS")
    print("=" * 100)

    try:

        result = extract_findings(
            dictation=dictation,
            structure=structure
        )

        print("\nExtraction completed successfully.")

    except Exception as e:

        print("\n[ERROR] EXTRACTION FAILED")
        print(f"Reason: {e}")

        traceback.print_exc()

        raise

    # ========================================================
    # 6. DETERMINISTIC HARD ROUTING
    # ========================================================

    print("\n" + "=" * 100)
    print("[6/7] APPLYING DETERMINISTIC ROUTING")
    print("=" * 100)

    try:

        # IMPORTANT:
        # extract_findings() already applies hard routing
        # internally.
        #
        # Therefore we DO NOT call it again here.
        #
        # This prevents the routing layer from being executed
        # twice.

        print(
            "Hard routing was already applied "
            "inside extract_findings()."
        )

        print("\nRunning final structural validation...")

        result = validate_result(result, structure)

        print("Structural validation PASSED.")

    except Exception as e:

        print("\n[ERROR] ROUTING / VALIDATION FAILED")
        print(f"Reason: {e}")

        traceback.print_exc()

        raise

    # ========================================================
    # 7. DEBUG EXTRACTION RESULT
    # ========================================================

    print("\n" + "=" * 100)
    print("[7/7] FINAL EXTRACTION RESULT")
    print("=" * 100)

    findings = result.findings

    print(f"\nTotal findings: {len(findings)}")

    print("\n" + "-" * 100)

    for index, finding in enumerate(findings, start=1):

        print(f"\nFINDING #{index}")
        print(f"  Finding       : {finding.finding}")
        print(f"  Negated       : {finding.negated}")
        print(f"  Laterality    : {finding.laterality}")
        print(f"  Measurement   : {finding.measurement}")
        print(f"  Parent        : {finding.parent_section}")
        print(f"  Target        : {finding.target_section}")
        print(f"  Action        : {finding.action}")

        print("-" * 80)

    # ========================================================
    # EXPECTED ROUTING CHECKS
    # ========================================================

    print("\n" + "=" * 100)
    print("EXPECTED SHOULDER ROUTING CHECK")
    print("=" * 100)

    expected_routes = [
        ("supraspinatus", "TENDONS", "SUPRASPINATUS"),
        ("subscapularis", "TENDONS", "SUBSCAPULARIS"),
        ("biceps", "TENDONS", "BICEPS BRACHII, LONG HEAD"),
        ("glenohumeral", "JOINTS", "GLENOHUMERAL JOINT"),
        ("acromioclavicular", "JOINTS", "ACROMIOCLAVICULAR JOINT"),
    ]

    for keyword, expected_parent, expected_target in expected_routes:

        matches = [
            f
            for f in findings
            if keyword.lower() in f.finding.lower()
        ]

        if not matches:

            print(f"[WARNING] No finding containing '{keyword}'")

            continue

        for finding in matches:

            actual_parent = finding.parent_section
            actual_target = finding.target_section

            if (
                actual_parent == expected_parent
                and actual_target == expected_target
            ):

                print(
                    f"[PASS] {keyword}"
                    f" -> {actual_parent}"
                    f" -> {actual_target}"
                )

            else:

                print(f"[FAIL] {keyword}")
                print(
                    f"       Expected: "
                    f"{expected_parent} -> {expected_target}"
                )
                print(
                    f"       Actual  : "
                    f"{actual_parent} -> {actual_target}"
                )

    # ========================================================
    # SAVE JSON RESULT
    # ========================================================

    output_file = "gemini_extraction_test.json"

    try:

        save_result(result, output_file)

        print("\n" + "=" * 100)
        print("RESULT SAVED")
        print("=" * 100)

        print(f"\nSaved structured extraction to:\n{output_file}")

    except Exception as e:

        print("\n[ERROR] Could not save result")
        print(f"Reason: {e}")

        traceback.print_exc()

        raise

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 100)
    print("DEBUG RUN COMPLETE")
    print("=" * 100)

    print(f"\nCase ID : {selected['case_id']}")
    print(f"Findings extracted : {len(result.findings)}")

    print("\nNext step: inspect the routing above.")
    print(
        "If routing is correct, move on to the "
        "template field editor."
    )