# src/impression_generator.py

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from gemini_retry import generate_content_with_retry

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Make sure it is present in your .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3-flash-preview"


# ---------------------------------------------------------
# Structured output
# ---------------------------------------------------------

class ImpressionResult(BaseModel):
    impression: str = Field(
        description=(
            "A concise radiology impression summarizing only the important "
            "patient-specific findings provided in the input. Do not add "
            "information that is not explicitly supported."
        )
    )


# ---------------------------------------------------------
# System prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a conservative radiology impression generator.

Your task is to generate ONLY the IMPRESSION portion of a radiology report
from validated patient-specific findings.

STRICT RULES:

1. Use ONLY information contained in the supplied findings.
2. NEVER introduce a new diagnosis, abnormality, measurement, anatomy,
   severity, laterality, or clinical interpretation that is not supported.
3. Do not use information from outside medical knowledge to add findings.
4. Preserve important:
   - laterality
   - anatomy
   - severity
   - measurements
   - negation
   - clinically meaningful qualifiers
5. Prioritize the most clinically important abnormalities.
6. Combine related findings when appropriate.
7. Do not repeat every minor normal finding.
8. Do not invent a normal impression such as "No acute abnormality"
   unless the supplied findings explicitly support it.
9. If a finding contains an explicit uncertainty or qualifier such as
   "may reflect" or "could represent", preserve that uncertainty.
10. Do not turn a possibility into a definite diagnosis.
11. Do not contradict any supplied finding.
12. Keep the impression concise.
13. Do not include the word "IMPRESSION:" in your response.
14. Do not use markdown.
15. Return only the final impression text.

The input findings may contain both positive and explicitly negated findings.
Negated findings should only be mentioned when clinically important or when
they materially clarify an important abnormality.
"""


# ---------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------

def build_impression_prompt(findings: List[dict]) -> str:
    """
    Convert validated FindingChange objects/dicts into a compact
    evidence list for the impression generator.
    """

    lines = []

    for i, finding in enumerate(findings, start=1):
        action = finding.get("action")

        # Routine unchanged template findings are not useful here.
        if action == "no_change":
            continue

        text = finding.get("finding", "").strip()

        if not text:
            continue

        negated = finding.get("negated", False)
        laterality = finding.get("laterality")
        measurement = finding.get("measurement")

        parts = [
            f"Finding {i}: {text}",
            f"Negated: {negated}",
        ]

        if laterality:
            parts.append(f"Laterality: {laterality}")

        if measurement:
            parts.append(f"Measurement: {measurement}")

        lines.append("\n".join(parts))

    if not lines:
        return """
There are no validated abnormal findings to summarize.

Generate a conservative impression based only on this information.
Do not invent abnormalities.
"""

    return f"""
Generate the radiology impression from the following validated findings.

VALIDATED FINDINGS:

{chr(10).join(lines)}

Remember:
- Use only these findings.
- Prioritize clinically important abnormalities.
- Do not invent information.
- Preserve uncertainty and negation where relevant.
- Keep the impression concise.
"""


# ---------------------------------------------------------
# Main generator
# ---------------------------------------------------------

def generate_impression(findings: List[dict]) -> str:
    """
    Generate a concise IMPRESSION from validated findings.

    Gemini is the primary model.
    Qwen is automatically used as fallback if Gemini
    encounters repeated transient failures.
    """

    prompt = build_impression_prompt(findings)

    response = generate_content_with_retry(
        client.models.generate_content,

        model=MODEL_NAME,
        contents=prompt,

        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=ImpressionResult,
        ),

        # ----------------------------------------------------
        # QWEN FALLBACK
        # ----------------------------------------------------

        fallback_prompt=prompt,
        fallback_system_instruction=SYSTEM_PROMPT,
        fallback_schema=ImpressionResult,
    )

    # --------------------------------------------------------
    # Qwen already returns ImpressionResult.
    # --------------------------------------------------------

    if isinstance(response, ImpressionResult):

        result = response

    else:

        result = response.parsed

        if result is None:

            raise ValueError(
                "Gemini did not return "
                "a structured impression."
            )

    impression = result.impression.strip()

    if not impression:

        raise ValueError(
            "Model returned an empty impression."
        )

    # --------------------------------------------------------
    # Safety cleanup
    # --------------------------------------------------------

    if impression.upper().startswith(
        "IMPRESSION:"
    ):

        impression = impression[
            len("IMPRESSION:"):
        ].strip()

    return impression

# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    test_findings = [
        {
            "finding": (
                "Moderate supraspinatus tendinosis with low-grade "
                "bursal-sided/intrasubstance partial-thickness insertional tear."
            ),
            "negated": False,
            "laterality": "right",
            "measurement": None,
            "target_section": "SUPRASPINATUS",
            "parent_section": "TENDONS",
            "action": "modify",
        },
        {
            "finding": (
                "No full-thickness rotator cuff tear or tendon retraction."
            ),
            "negated": True,
            "laterality": "right",
            "measurement": None,
            "target_section": "SUPRASPINATUS",
            "parent_section": "TENDONS",
            "action": "modify",
        },
        {
            "finding": (
                "Moderate glenohumeral joint effusion with distention "
                "of the subscapularis recess."
            ),
            "negated": False,
            "laterality": "right",
            "measurement": None,
            "target_section": "GLENOHUMERAL JOINT",
            "parent_section": "JOINTS",
            "action": "modify",
        },
        {
            "finding": (
                "Moderate acromioclavicular osteoarthrosis with inferior "
                "osteophytic spurring resulting in mild subacromial outlet narrowing."
            ),
            "negated": False,
            "laterality": "right",
            "measurement": None,
            "target_section": "ACROMIOCLAVICULAR JOINT",
            "parent_section": "JOINTS",
            "action": "modify",
        },
        {
            "finding": (
                "Moderate subacromial-subdeltoid bursitis."
            ),
            "negated": False,
            "laterality": "right",
            "measurement": None,
            "target_section": None,
            "parent_section": None,
            "action": "insert",
        },
    ]

    impression = generate_impression(test_findings)

    print("\nIMPRESSION:")
    print(impression)