# src/validator.py

from typing import List, Dict, Any, Optional


# =========================================================
# Basic helpers
# =========================================================

def normalize_name(name: Optional[str]) -> str:
    """
    Normalize section names for comparison.
    """
    if name is None:
        return ""

    return " ".join(str(name).upper().split())


def normalize_text(text: Optional[str]) -> str:
    """
    Basic text normalization for validation.
    Does NOT modify the actual report.
    """
    if text is None:
        return ""

    return " ".join(str(text).split())


# =========================================================
# Template section lookup
# =========================================================

def find_section(
    sections: List[Dict[str, Any]],
    target_name: str,
    parent_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find a section in the parsed template.

    If parent_name is supplied, first look for the target
    specifically under that parent.
    """

    target = normalize_name(target_name)
    parent = normalize_name(parent_name)

    # -----------------------------------------------------
    # First: exact parent -> child search
    # -----------------------------------------------------

    if parent:
        for section in sections:
            section_name = normalize_name(section.get("name"))

            if section_name == parent:
                for child in section.get("children", []):
                    if normalize_name(child.get("name")) == target:
                        return child

                # Parent was found but child wasn't.
                # Continue recursively in case hierarchy is deeper.
                result = find_section(
                    section.get("children", []),
                    target_name,
                    parent_name,
                )

                if result is not None:
                    return result

        # Don't accidentally match an unrelated root section
        # when a parent was explicitly supplied.
        return None

    # -----------------------------------------------------
    # No parent supplied: search recursively
    # -----------------------------------------------------

    for section in sections:

        if normalize_name(section.get("name")) == target:
            return section

        result = find_section(
            section.get("children", []),
            target_name,
            None,
        )

        if result is not None:
            return result

    return None


# =========================================================
# Finding validation
# =========================================================

def validate_finding(
    finding: Dict[str, Any],
    template_sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate one FindingChange object.

    Returns:

    {
        "valid": True/False,
        "errors": [...],
        "warnings": [...]
    }
    """

    errors = []
    warnings = []

    action = finding.get("action")
    text = normalize_text(finding.get("finding"))
    target = finding.get("target_section")
    parent = finding.get("parent_section")
    negated = finding.get("negated")

    # -----------------------------------------------------
    # Required finding text
    # -----------------------------------------------------

    if not text:
        errors.append("Finding text is empty.")

    # -----------------------------------------------------
    # Action validation
    # -----------------------------------------------------

    valid_actions = {
        "no_change",
        "modify",
        "insert",
    }

    if action not in valid_actions:
        errors.append(
            f"Invalid action: {action}. "
            f"Expected one of {sorted(valid_actions)}."
        )

    # -----------------------------------------------------
    # Negation validation
    # -----------------------------------------------------

    if not isinstance(negated, bool):
        errors.append(
            "'negated' must be a boolean."
        )

    # -----------------------------------------------------
    # NO CHANGE
    # -----------------------------------------------------

    if action == "no_change":

        if target is None:
            warnings.append(
                "no_change finding has no target section."
            )

        # no_change should never cause an edit.
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # -----------------------------------------------------
    # MODIFY
    # -----------------------------------------------------

    if action == "modify":

        if not target:
            errors.append(
                "modify action requires target_section."
            )
        else:

            section = find_section(
                template_sections,
                target,
                parent,
            )

            if section is None:
                errors.append(
                    f"Target section '{target}' "
                    f"with parent '{parent}' was not found "
                    f"in the template."
                )

        # A modify finding should have a target.
        if target and parent:

            parent_section = find_section(
                template_sections,
                parent,
                None,
            )

            if parent_section is None:
                errors.append(
                    f"Parent section '{parent}' "
                    f"was not found in the template."
                )

    # -----------------------------------------------------
    # INSERT
    # -----------------------------------------------------

    if action == "insert":

        # Insertions must not pretend to target an existing
        # field. They are handled separately.
        if target is not None:
            errors.append(
                "insert action must have target_section=None."
            )

        if parent is not None:
            errors.append(
                "insert action must have parent_section=None."
            )

        if not text:
            errors.append(
                "insert action requires non-empty finding text."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# =========================================================
# Validate all findings
# =========================================================

def validate_findings(
    findings: List[Dict[str, Any]],
    template_sections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validate all extracted findings.

    Returns:

    {
        "valid": True/False,
        "valid_findings": [...],
        "invalid_findings": [...],
        "errors": [...],
        "warnings": [...]
    }
    """

    errors = []
    warnings = []

    valid_findings = []
    invalid_findings = []

    if not isinstance(findings, list):
        return {
            "valid": False,
            "valid_findings": [],
            "invalid_findings": [],
            "errors": ["Findings must be a list."],
            "warnings": [],
        }

    for index, finding in enumerate(findings):

        if not isinstance(finding, dict):
            error = f"Finding {index} is not a dictionary."

            errors.append(error)

            invalid_findings.append({
                "index": index,
                "finding": finding,
                "errors": [error],
            })

            continue

        result = validate_finding(
            finding,
            template_sections,
        )

        if result["valid"]:

            valid_findings.append(finding)

        else:

            invalid_findings.append({
                "index": index,
                "finding": finding,
                "errors": result["errors"],
            })

            for error in result["errors"]:
                errors.append(
                    f"Finding {index}: {error}"
                )

        for warning in result["warnings"]:
            warnings.append(
                f"Finding {index}: {warning}"
            )

    return {
        "valid": len(errors) == 0,
        "valid_findings": valid_findings,
        "invalid_findings": invalid_findings,
        "errors": errors,
        "warnings": warnings,
    }


# =========================================================
# Duplicate finding detection
# =========================================================

def find_duplicate_findings(
    findings: List[Dict[str, Any]]
) -> List[List[int]]:
    """
    Detect obvious duplicate findings.

    Returns groups of duplicate indexes.

    Example:

    [[0, 3], [2, 5]]
    """

    seen = {}
    duplicates = []

    for index, finding in enumerate(findings):

        text = normalize_text(
            finding.get("finding")
        ).lower()

        if not text:
            continue

        key = (
            text,
            normalize_name(
                finding.get("target_section")
            ),
            normalize_name(
                finding.get("parent_section")
            ),
            finding.get("negated"),
        )

        if key in seen:

            # First duplicate occurrence.
            existing_index = seen[key]

            # Check whether we already created a group.
            found_group = None

            for group in duplicates:
                if existing_index in group:
                    found_group = group
                    break

            if found_group is not None:
                found_group.append(index)

            else:
                duplicates.append(
                    [existing_index, index]
                )

        else:
            seen[key] = index

    return duplicates


# =========================================================
# Template preservation validation
# =========================================================

def flatten_sections(
    sections: List[Dict[str, Any]],
    parent_name: Optional[str] = None,
) -> Dict[tuple, str]:
    """
    Flatten hierarchical template sections.

    Returns:
        {
            (parent_name, section_name): content
        }
    """

    result = {}

    for section in sections:
        name = section.get("name")

        if name is not None:
            key = (
                normalize_name(parent_name),
                normalize_name(name),
            )

            result[key] = normalize_text(
                section.get("content", "")
            )

            children = section.get("children", [])

            if children:
                result.update(
                    flatten_sections(
                        children,
                        name,
                    )
                )

        else:
            # Unlabelled/root content
            key = (
                normalize_name(parent_name),
                "",
            )

            result[key] = normalize_text(
                section.get("content", "")
            )

    return result


def validate_untouched_sections(
    original_sections: List[Dict[str, Any]],
    updated_sections: List[Dict[str, Any]],
    edited_fields: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify that sections which were NOT intentionally edited
    remain unchanged.
    """

    errors = []
    warnings = []

    original = flatten_sections(original_sections)
    updated = flatten_sections(updated_sections)

    intentionally_edited = set()

    for field in edited_fields:

        parent = normalize_name(
            field.get("parent_section")
        )

        section = normalize_name(
            field.get("section_name")
        )

        intentionally_edited.add(
            (parent, section)
        )

    # -----------------------------------------------------
    # Check original sections
    # -----------------------------------------------------

    for key, original_content in original.items():

        if key in intentionally_edited:
            continue

        if key not in updated:

            errors.append(
                f"Untouched section disappeared: {key}"
            )

            continue

        updated_content = updated[key]

        if original_content != updated_content:

            errors.append(
                f"Untouched section changed: {key}"
            )

    # -----------------------------------------------------
    # Check for unexpected new sections
    # -----------------------------------------------------

    for key in updated:

        if key not in original:

            warnings.append(
                f"New section detected: {key}"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_edited_fields(
    edited_fields: List[Dict[str, Any]],
) -> Dict[str, Any]:

    errors = []
    warnings = []

    for field in edited_fields:

        section_name = field.get("section_name")

        original = normalize_text(
            field.get("original_content")
        ).lower()

        edited = normalize_text(
            field.get("edited_content")
        ).lower()

        findings = field.get("findings", [])

        if not edited:
            errors.append(
                f"{section_name}: edited content is empty."
            )
            continue

        for finding in findings:

            finding_text = normalize_text(
                finding.get("finding")
            ).lower()

            if not finding_text:
                continue

            # We don't require the entire finding verbatim,
            # because the editor may naturally rewrite it.
            #
            # Instead check important anatomical terms.
            target = normalize_name(
                finding.get("target_section")
            ).lower()

            if target and target not in edited:

                warnings.append(
                    f"{section_name}: target anatomy "
                    f"'{target}' not found in edited content."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# =========================================================
# Report structure validation
# =========================================================

def validate_report_structure(
    report: str,
) -> Dict[str, Any]:
    """
    Validate the final report format.

    Expected:

    FINDINGS:
    ...

    IMPRESSION:
    ...
    """

    errors = []
    warnings = []

    if not isinstance(report, str):
        return {
            "valid": False,
            "errors": ["Report must be a string."],
            "warnings": [],
        }

    text = report.strip()

    if not text:
        errors.append("Report is empty.")

        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    # -----------------------------------------------------
    # Normalize lines for exact header detection
    # -----------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    # -----------------------------------------------------
    # Required headers
    # -----------------------------------------------------

    if not text.startswith("FINDINGS:"):
        errors.append(
            "Report must start with 'FINDINGS:'."
        )

    # IMPORTANT:
    # Use exact-line matching.
    #
    # "OTHER FINDINGS:" must NOT count as "FINDINGS:".
    # "IMPRESSION:" must also be detected only as its
    # own header line.
    # -----------------------------------------------------

    findings_headers = [
        line
        for line in lines
        if line == "FINDINGS:"
    ]

    impression_headers = [
        line
        for line in lines
        if line == "IMPRESSION:"
    ]

    # -----------------------------------------------------
    # Missing headers
    # -----------------------------------------------------

    if not findings_headers:
        errors.append(
            "Report is missing 'FINDINGS:'."
        )

    if not impression_headers:
        errors.append(
            "Report is missing 'IMPRESSION:'."
        )

    # -----------------------------------------------------
    # Exactly one FINDINGS header
    # -----------------------------------------------------

    if len(findings_headers) != 1:
        errors.append(
            "Report must contain exactly one 'FINDINGS:' header."
        )

    # -----------------------------------------------------
    # Exactly one IMPRESSION header
    # -----------------------------------------------------

    if len(impression_headers) != 1:
        errors.append(
            "Report must contain exactly one 'IMPRESSION:' header."
        )

    # -----------------------------------------------------
    # Impression must not be empty
    # -----------------------------------------------------

    if impression_headers:

        impression_index = next(
            i
            for i, line in enumerate(lines)
            if line == "IMPRESSION:"
        )

        impression = "\n".join(
            lines[impression_index + 1:]
        ).strip()

        if not impression:
            errors.append(
                "IMPRESSION section is empty."
            )

    # -----------------------------------------------------
    # Return validation result
    # -----------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
# =========================================================
# Complete validation
# =========================================================

def validate_pipeline_output(
    findings: List[Dict[str, Any]],
    template_sections: List[Dict[str, Any]],
    report: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all deterministic validation checks.
    """

    finding_result = validate_findings(
        findings,
        template_sections,
    )

    duplicates = find_duplicate_findings(
        findings
    )

    warnings = list(
        finding_result["warnings"]
    )

    if duplicates:

        for group in duplicates:
            warnings.append(
                f"Possible duplicate findings at indexes: {group}"
            )

    report_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    if report is not None:

        report_result = validate_report_structure(
            report
        )

    errors = (
        finding_result["errors"]
        + report_result["errors"]
    )

    warnings.extend(
        report_result["warnings"]
    )

    return {
        "valid": len(errors) == 0,
        "finding_validation": finding_result,
        "report_validation": report_result,
        "errors": errors,
        "warnings": warnings,
    }


# =========================================================
# Pretty printer
# =========================================================

def print_validation_result(result: Dict[str, Any]):

    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    print(
        f"\nValid: {result['valid']}"
    )

    if result["errors"]:

        print("\nERRORS:")

        for error in result["errors"]:
            print(f"  ❌ {error}")

    else:
        print("\nERRORS:")
        print("  None")

    if result["warnings"]:

        print("\nWARNINGS:")

        for warning in result["warnings"]:
            print(f"  ⚠️ {warning}")

    else:
        print("\nWARNINGS:")
        print("  None")

    print("=" * 60)


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    from template_structure import build_template_structure

    # -----------------------------------------------------
    # Small shoulder template for testing
    # -----------------------------------------------------

    test_template = """
FINDINGS:

TENDONS:
SUPRASPINATUS: The tendon is intact. No tear.
INFRASPINATUS: The tendon is intact.
SUBSCAPULARIS: The tendon is intact.

JOINTS:
GLENOHUMERAL JOINT: No joint effusion. The joint spaces are maintained.

BONES: No acute fracture or focal osseous lesion.

IMPRESSION:
Normal MRI of the shoulder.
"""

    structure = build_template_structure(
        test_template
    )

    template_sections = structure["findings"]

    # -----------------------------------------------------
    # Valid findings
    # -----------------------------------------------------

    test_findings = [
        {
            "finding": (
                "Moderate supraspinatus tendinosis with "
                "low-grade partial-thickness tearing."
            ),
            "negated": False,
            "laterality": "right",
            "measurement": None,
            "target_section": "SUPRASPINATUS",
            "parent_section": "TENDONS",
            "action": "modify",
        },
        {
            "finding": "Moderate glenohumeral joint effusion.",
            "negated": False,
            "laterality": "right",
            "measurement": None,
            "target_section": "GLENOHUMERAL JOINT",
            "parent_section": "JOINTS",
            "action": "modify",
        },
        {
            "finding": "Moderate subacromial-subdeltoid bursitis.",
            "negated": False,
            "laterality": "right",
            "measurement": None,
            "target_section": None,
            "parent_section": None,
            "action": "insert",
        },
    ]

    result = validate_pipeline_output(
        findings=test_findings,
        template_sections=template_sections,
    )

    print_validation_result(result)

    # -----------------------------------------------------
    # Test malformed finding
    # -----------------------------------------------------

    bad_finding = {
        "finding": "Some finding",
        "negated": False,
        "laterality": None,
        "measurement": None,
        "target_section": "DOES NOT EXIST",
        "parent_section": "TENDONS",
        "action": "modify",
    }

    bad_result = validate_pipeline_output(
        findings=[bad_finding],
        template_sections=template_sections,
    )

    print("\nTesting invalid finding:")

    print_validation_result(
        bad_result
    )

    # -----------------------------------------------------
    # Test untouched-section preservation
    # -----------------------------------------------------

    original_structure = build_template_structure(
        test_template
    )

    original_sections = original_structure["findings"]

    # Simulate expected edited template
    edited_template = """
FINDINGS:

TENDONS:
SUPRASPINATUS: Moderate supraspinatus tendinosis with low-grade partial-thickness tearing.
INFRASPINATUS: The tendon is intact.
SUBSCAPULARIS: The tendon is intact.

JOINTS:
GLENOHUMERAL JOINT: Moderate glenohumeral joint effusion. The joint spaces are maintained.

BONES: No acute fracture or focal osseous lesion.

IMPRESSION:
Abnormal MRI of the shoulder.
"""

    edited_structure = build_template_structure(
        edited_template
    )

    edited_sections = edited_structure["findings"]

    edited_fields = [
        {
            "section_name": "SUPRASPINATUS",
            "parent_section": "TENDONS",
        },
        {
            "section_name": "GLENOHUMERAL JOINT",
            "parent_section": "JOINTS",
        },
    ]

    preservation_result = validate_untouched_sections(
        original_sections=original_sections,
        updated_sections=edited_sections,
        edited_fields=edited_fields,
    )

    print("\nTesting untouched-section preservation:")

    print(
        f"Valid: {preservation_result['valid']}"
    )

    if preservation_result["errors"]:

        print("\nErrors:")

        for error in preservation_result["errors"]:
            print(f"  ❌ {error}")

    else:

        print("  ✅ No untouched sections were changed.")