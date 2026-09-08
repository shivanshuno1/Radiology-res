
# src/template_editor.py

from typing import Any, Dict, List, Optional
import re

from template_structure import PARENT_SECTIONS


# =========================================================
# Normalization
# =========================================================

def normalize_name(
    name: Optional[str],
) -> str:
    """
    Normalize section names for comparison only.

    The original template is never modified here.
    """

    if name is None:
        return ""

    return " ".join(
        str(name).upper().split()
    )


# =========================================================
# Section header detection
# =========================================================

def is_section_header(
    line: str,
) -> bool:
    """
    Detect a standalone section header.

    Examples:

        BONES:
        TENDONS:
        SUPRASPINATUS:
        GLENOHUMERAL JOINT:

    FINDINGS and IMPRESSION are excluded.

    Inline headers such as:

        SUPRASPINATUS: The tendon is intact.

    are handled separately by find_section_spans().
    """

    stripped = line.strip()

    if not stripped:
        return False

    if not stripped.endswith(":"):
        return False

    name = stripped[:-1].strip()

    if not name:
        return False

    if normalize_name(name) in {
        "FINDINGS",
        "IMPRESSION",
    }:
        return False

    return bool(
        re.match(
            r"^[A-Za-z0-9][A-Za-z0-9 /&(),.'\-]*$",
            name,
        )
    )


# =========================================================
# Inline section detection
# =========================================================

def parse_inline_section_header(
    line: str,
):
    """
    Detect an inline section.

    Example:

        SUPRASPINATUS: The tendon is intact.

    Returns:

        (section_name, content)

    or:

        (None, None)

    This is deliberately conservative so that ordinary
    prose containing ':' is not incorrectly interpreted
    as a section.
    """

    stripped = line.strip()

    if not stripped:
        return None, None

    if ":" not in stripped:
        return None, None

    name, content = stripped.split(
        ":",
        1,
    )

    name = name.strip()
    content = content.strip()

    if not name:
        return None, None

    if normalize_name(name) in {
        "FINDINGS",
        "IMPRESSION",
    }:
        return None, None

    if not re.match(
        r"^[A-Za-z0-9][A-Za-z0-9 /&(),.'\-]*$",
        name,
    ):
        return None, None

    return name, content


# =========================================================
# Find raw section spans
# =========================================================

def find_section_spans(
    template: str,
) -> List[Dict[str, Any]]:
    """
    Find section spans in the raw template.

    Supports both:

        SECTION:
        content

    and:

        SECTION: content

    The original template text is never modified.

    Returned fields:

        name
        start_line
        content_start_line
        end_line
        inline_content
        parent_name
    """

    lines = template.splitlines(
        keepends=True
    )

    spans: List[Dict[str, Any]] = []

    current = None

    for index, line in enumerate(lines):

        stripped = line.strip()

        # -------------------------------------------------
        # FINDINGS
        # -------------------------------------------------

        if normalize_name(
            stripped.rstrip(":")
        ) == "FINDINGS":

            continue

        # -------------------------------------------------
        # IMPRESSION
        # -------------------------------------------------

        if normalize_name(
            stripped.rstrip(":")
        ) == "IMPRESSION":

            if current is not None:

                current["end_line"] = index

                spans.append(
                    current
                )

                current = None

            # Nothing after IMPRESSION belongs to
            # FINDINGS sections.
            break

        # -------------------------------------------------
        # Detect section
        # -------------------------------------------------

        section_name = None
        inline_content = None

        # -------------------------------------------------
        # Case 1: standalone header
        #
        # SECTION:
        # -------------------------------------------------

        if is_section_header(line):

            section_name = stripped[:-1].strip()

        # -------------------------------------------------
        # Case 2: inline header
        #
        # SECTION: content
        # -------------------------------------------------

        else:

            (
                possible_name,
                possible_content,
            ) = parse_inline_section_header(
                line
            )

            if possible_name is not None:

                section_name = possible_name
                inline_content = possible_content

        # -------------------------------------------------
        # New section
        # -------------------------------------------------

        if section_name is not None:

            if current is not None:

                current["end_line"] = index

                spans.append(
                    current
                )

            current = {
                "name": section_name,
                "start_line": index,

                # For inline sections the content is
                # located on the same physical line.
                #
                # For standalone sections it begins
                # on the following line.
                "content_start_line": (
                    index
                    if inline_content is not None
                    else index + 1
                ),

                "end_line": None,

                "inline_content": inline_content,

                "parent_name": None,
            }

    # -----------------------------------------------------
    # Close final section
    # -----------------------------------------------------

    if current is not None:

        current["end_line"] = len(lines)

        spans.append(
            current
        )

    # -----------------------------------------------------
    # Attach semantic hierarchy
    # -----------------------------------------------------

    return _attach_semantic_parents(
        spans
    )


# =========================================================
# Semantic hierarchy
# =========================================================

def _is_known_parent(
    section_name: str,
) -> bool:
    """
    Check whether a section is a known semantic parent.
    """

    normalized = normalize_name(
        section_name
    )

    return normalized in {
        normalize_name(name)
        for name in PARENT_SECTIONS.keys()
    }


def _is_known_child_of(
    child_name: str,
    parent_name: str,
) -> bool:
    """
    Check whether child_name belongs to parent_name
    according to PARENT_SECTIONS from template_structure.py.
    """

    parent = normalize_name(
        parent_name
    )

    child = normalize_name(
        child_name
    )

    children = PARENT_SECTIONS.get(
        parent,
        set(),
    )

    return child in {
        normalize_name(name)
        for name in children
    }


def _attach_semantic_parents(
    spans: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Assign semantic parent_name to nested sections.

    This does NOT use indentation.

    Example:

        TENDONS:
        SUPRASPINATUS: ...
        INFRASPINATUS: ...
        SUBSCAPULARIS: ...

    becomes:

        TENDONS -> None
        SUPRASPINATUS -> TENDONS
        INFRASPINATUS -> TENDONS
        SUBSCAPULARIS -> TENDONS

    The hierarchy comes from PARENT_SECTIONS.
    """

    current_parent = None

    for span in spans:

        name = span["name"]

        # -------------------------------------------------
        # Known parent
        # -------------------------------------------------

        if _is_known_parent(name):

            span["parent_name"] = None

            current_parent = name

            continue

        # -------------------------------------------------
        # Known child
        # -------------------------------------------------

        if (
            current_parent is not None
            and _is_known_child_of(
                name,
                current_parent,
            )
        ):

            span["parent_name"] = (
                current_parent
            )

            continue

        # -------------------------------------------------
        # Ordinary top-level section
        # -------------------------------------------------

        span["parent_name"] = None

        current_parent = None

    return spans


# =========================================================
# Find exact section
# =========================================================

def find_section_span(
    spans: List[Dict[str, Any]],
    target_name: str,
    parent_name: Optional[str] = None,
):
    """
    Find the physical span corresponding to a target section.

    If parent_name is supplied, BOTH target and parent must
    match.

    This prevents editing the wrong duplicate section.
    """

    target = normalize_name(
        target_name
    )

    parent = normalize_name(
        parent_name
    )

    # -----------------------------------------------------
    # Parent + target
    # -----------------------------------------------------

    if parent:

        matches = []

        for span in spans:

            if normalize_name(
                span["name"]
            ) != target:

                continue

            actual_parent = normalize_name(
                span.get(
                    "parent_name"
                )
            )

            if actual_parent == parent:

                matches.append(
                    span
                )

        if len(matches) == 1:

            return matches[0]

        if len(matches) > 1:

            raise ValueError(
                f"Multiple sections found for "
                f"{parent_name} -> {target_name}."
            )

        # Never fall back to target-only when a parent
        # was explicitly requested.
        return None

    # -----------------------------------------------------
    # Target only
    # -----------------------------------------------------

    matches = [
        span
        for span in spans
        if normalize_name(
            span["name"]
        ) == target
    ]

    if not matches:

        return None

    if len(matches) == 1:

        return matches[0]

    raise ValueError(
        f"Multiple sections found with name "
        f"'{target_name}'. Parent section is required."
    )


# =========================================================
# Extract indentation
# =========================================================

def _get_content_indentation(
    lines: List[str],
    start: int,
    end: int,
) -> str:
    """
    Determine the indentation used by original content.
    """

    for index in range(
        start,
        end,
    ):

        line = lines[index]

        if not line.strip():
            continue

        return line[
            :len(line)
            - len(line.lstrip())
        ]

    return ""


# =========================================================
# Replace section content
# =========================================================

def replace_section_content(
    template: str,
    target_name: str,
    new_content: str,
    parent_name: Optional[str] = None,
) -> str:
    """
    Replace ONLY the content of one existing section.

    Supports:

        SECTION:
        content

    and:

        SECTION: content

    The section header and all unrelated template text
    remain untouched.
    """

    lines = template.splitlines(
        keepends=True
    )

    spans = find_section_spans(
        template
    )

    span = find_section_span(
        spans,
        target_name,
        parent_name,
    )

    if span is None:

        raise ValueError(
            f"Could not find section: "
            f"{parent_name + ' -> ' if parent_name else ''}"
            f"{target_name}"
        )

    start = span[
        "start_line"
    ]

    end = span[
        "end_line"
    ]

    new_content = str(
        new_content
    ).strip()

    # =====================================================
    # CASE 1: INLINE SECTION
    #
    # Example:
    #
    # SUPRASPINATUS: The tendon is intact.
    #
    # Only the text after ':' is replaced.
    # =====================================================

    if span.get(
        "inline_content"
    ) is not None:

        original_line = lines[start]

        colon_index = original_line.find(":")

        if colon_index == -1:

            raise ValueError(
                f"Invalid inline section: "
                f"{target_name}"
            )

        header = original_line[
            :colon_index + 1
        ]

        # Preserve the original newline.
        if original_line.endswith(
            "\r\n"
        ):

            newline = "\r\n"

        elif original_line.endswith(
            "\n"
        ):

            newline = "\n"

        elif original_line.endswith(
            "\r"
        ):

            newline = "\r"

        else:

            newline = ""

        if new_content:

            lines[start] = (
                header
                + " "
                + new_content
                + newline
            )

        else:

            lines[start] = (
                header
                + newline
            )

        return "".join(
            lines
        )

    # =====================================================
    # CASE 2: STANDALONE SECTION
    #
    # Example:
    #
    # SUPRASPINATUS:
    # The tendon is intact.
    # =====================================================

    content_start = span[
        "content_start_line"
    ]

    indentation = (
        _get_content_indentation(
            lines,
            content_start,
            end,
        )
    )

    # -----------------------------------------------------
    # Preserve trailing blank lines
    # -----------------------------------------------------

    trailing_blank_lines = 0

    for index in range(
        end - 1,
        content_start - 1,
        -1,
    ):

        if lines[index].strip() == "":

            trailing_blank_lines += 1

        else:

            break

    # -----------------------------------------------------
    # Build replacement
    # -----------------------------------------------------

    replacement = []

    content_lines = new_content.splitlines()

    for line in content_lines:

        if line.strip():

            replacement.append(
                indentation
                + line.strip()
                + "\n"
            )

    # Restore original trailing blank lines.
    for _ in range(
        trailing_blank_lines
    ):

        replacement.append(
            "\n"
        )

    # -----------------------------------------------------
    # Replace only content
    # -----------------------------------------------------

    updated_lines = (
        lines[:content_start]
        + replacement
        + lines[end:]
    )

    return "".join(
        updated_lines
    )


# =========================================================
# Normalize inserted findings
# =========================================================

def _normalize_inserted_finding(
    finding: Any,
) -> str:
    """
    Normalize an inserted finding for safe comparison.

    This does NOT medically rewrite the finding.

    It only:
        - converts to string
        - strips surrounding whitespace
        - collapses repeated whitespace
    """

    if finding is None:

        return ""

    text = str(
        finding
    ).strip()

    if not text:

        return ""

    return re.sub(
        r"\s+",
        " ",
        text,
    )


# =========================================================
# Deduplicate inserted findings
# =========================================================

def _deduplicate_inserted_findings(
    insertions: List[Dict[str, Any]],
) -> List[str]:
    """
    Deduplicate insertion findings while preserving
    their original order.

    Comparison is case-insensitive and whitespace-normalized.

    The actual finding text is otherwise preserved.
    """

    output: List[str] = []

    seen = set()

    for edit in insertions:

        finding = edit.get(
            "edited_content"
        )

        if finding is None:

            finding = edit.get(
                "finding"
            )

        finding = _normalize_inserted_finding(
            finding
        )

        if not finding:

            continue

        key = finding.lower()

        if key in seen:

            continue

        seen.add(
            key
        )

        output.append(
            finding
        )

    return output


# =========================================================
# Insert OTHER FINDINGS before IMPRESSION
# =========================================================

def insert_other_findings(
    template: str,
    insertions: List[Dict[str, Any]],
) -> str:
    """
    Insert unlabelled findings into a deterministic
    OTHER FINDINGS section immediately before IMPRESSION.

    Existing template content is preserved.

    Behavior:

        If insertions are empty:
            return template unchanged.

        If OTHER FINDINGS already exists:
            append new findings to that section.

        If OTHER FINDINGS does not exist:
            create it immediately before IMPRESSION.

    This function does NOT create an OTHER FINDINGS section
    unless there is at least one valid insertion.
    """

    findings = _deduplicate_inserted_findings(
        insertions
    )

    if not findings:

        return template

    lines = template.splitlines(
        keepends=True
    )

    # -----------------------------------------------------
    # Find IMPRESSION
    # -----------------------------------------------------

    impression_index = None

    for index, line in enumerate(lines):

        if normalize_name(
            line.strip().rstrip(":")
        ) == "IMPRESSION":

            impression_index = index

            break

    if impression_index is None:

        raise ValueError(
            "Could not find IMPRESSION section "
            "while inserting OTHER FINDINGS."
        )

    # -----------------------------------------------------
    # Check whether OTHER FINDINGS already exists.
    #
    # We deliberately inspect only the FINDINGS portion
    # before IMPRESSION.
    # -----------------------------------------------------

    existing_other_index = None

    for index in range(
        impression_index
    ):

        if normalize_name(
            lines[index].strip().rstrip(":")
        ) == "OTHER FINDINGS":

            existing_other_index = index

            break

    # =====================================================
    # CASE 1:
    # OTHER FINDINGS already exists
    # =====================================================

    if existing_other_index is not None:

        # Find the next section after OTHER FINDINGS.
        next_section_index = impression_index

        for index in range(
            existing_other_index + 1,
            impression_index,
        ):

            if is_section_header(
                lines[index]
            ):

                next_section_index = index

                break

            (
                inline_name,
                _,
            ) = parse_inline_section_header(
                lines[index]
            )

            if inline_name is not None:

                next_section_index = index

                break

        # Determine indentation from existing content.
        indentation = _get_content_indentation(
            lines,
            existing_other_index + 1,
            next_section_index,
        )

        # Insert immediately before the next section.
        replacement = []

        for finding in findings:

            replacement.append(
                indentation
                + finding
                + "\n"
            )

        updated_lines = (
            lines[:next_section_index]
            + replacement
            + lines[next_section_index:]
        )

        return "".join(
            updated_lines
        )

    # =====================================================
    # CASE 2:
    # OTHER FINDINGS does not exist
    #
    # Insert immediately before IMPRESSION.
    # =====================================================

    insertion = []

    # Ensure separation from the previous section.
    if (
        impression_index > 0
        and lines[impression_index - 1].strip()
    ):

        insertion.append(
            "\n"
        )

    insertion.append(
        "OTHER FINDINGS:\n"
    )

    for finding in findings:

        insertion.append(
            finding
            + "\n"
        )

    # Blank line between OTHER FINDINGS and IMPRESSION.
    insertion.append(
        "\n"
    )

    updated_lines = (
        lines[:impression_index]
        + insertion
        + lines[impression_index:]
    )

    return "".join(
        updated_lines
    )


# =========================================================
# Apply multiple edits
# =========================================================

def apply_section_edits(
    template: str,
    edits: List[Dict[str, Any]],
) -> str:
    """
    Apply multiple section modifications safely.

    Each edit may contain:

        target_section
        parent_section
        edited_content
        action

    Supported actions:

        modify
        replace
        insert

    For insertions, the finding is placed into a deterministic
    OTHER FINDINGS section immediately before IMPRESSION.

    Existing section modifications are applied first.
    Insertions are applied afterward.

    This prevents inserted content from interfering with
    section-span calculations.
    """

    result = template

    deduplicated = {}

    insertions: List[Dict[str, Any]] = []

    # =====================================================
    # Deduplicate / separate insertions
    # =====================================================

    for edit in edits:

        action = edit.get(
            "action",
            "modify",
        )

        # -------------------------------------------------
        # Insertions
        # -------------------------------------------------

        if action == "insert":

            insertions.append(
                edit
            )

            continue

        # -------------------------------------------------
        # Existing section edits
        # -------------------------------------------------

        target = edit.get(
            "target_section"
        )

        if not target:

            continue

        parent = edit.get(
            "parent_section"
        )

        key = (
            normalize_name(parent),
            normalize_name(target),
        )

        deduplicated[key] = edit

    # =====================================================
    # Apply existing section edits
    # =====================================================

    for edit in deduplicated.values():

        action = edit.get(
            "action",
            "modify",
        )

        if action not in {
            "modify",
            "replace",
        }:

            continue

        target = edit.get(
            "target_section"
        )

        parent = edit.get(
            "parent_section"
        )

        content = edit.get(
            "edited_content",
            "",
        )

        if not target:

            continue

        result = replace_section_content(
            template=result,
            target_name=target,
            parent_name=parent,
            new_content=content,
        )

    # =====================================================
    # Apply insertions LAST
    # =====================================================

    if insertions:

        result = insert_other_findings(
            template=result,
            insertions=insertions,
        )

    return result


# =========================================================
# Section existence
# =========================================================

def section_exists(
    template: str,
    target_name: str,
    parent_name: Optional[str] = None,
) -> bool:
    """
    Return True if a requested section exists.
    """

    spans = find_section_spans(
        template
    )

    try:

        return (
            find_section_span(
                spans,
                target_name,
                parent_name,
            )
            is not None
        )

    except ValueError:

        return False


# =========================================================
# Debugging helper
# =========================================================

def print_section_tree(
    template: str,
):
    """
    Print the section hierarchy detected by the editor.
    """

    spans = find_section_spans(
        template
    )

    print(
        "\nDetected template sections:"
    )

    for span in spans:

        parent = span.get(
            "parent_name"
        )

        if parent:

            print(
                f"  {parent} -> "
                f"{span['name']}"
            )

        else:

            print(
                f"  {span['name']}"
            )


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    template = """FINDINGS:
TENDONS:
SUPRASPINATUS:
The tendon is intact.

INFRASPINATUS:
The tendon is intact with normal signal intensity.

SUBSCAPULARIS:
The tendon is intact.

JOINTS:
GLENOHUMERAL JOINT:
No joint effusion.

ACROMIOCLAVICULAR JOINT:
The joint is normal.

BONES:
No acute fracture.

IMPRESSION:
No acute abnormality.
"""

    print_section_tree(
        template
    )

    # -----------------------------------------------------
    # Test existing section replacement
    # -----------------------------------------------------

    edited = replace_section_content(
        template=template,
        target_name="SUPRASPINATUS",
        parent_name="TENDONS",
        new_content=(
            "Moderate supraspinatus tendinosis "
            "with partial-thickness tearing."
        ),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ORIGINAL:"
    )

    print(template)

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EDITED:"
    )

    print(edited)

    # -----------------------------------------------------
    # Test another nested section
    # -----------------------------------------------------

    edited_2 = replace_section_content(
        template=edited,
        target_name="GLENOHUMERAL JOINT",
        parent_name="JOINTS",
        new_content=(
            "Moderate glenohumeral joint effusion."
        ),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SECOND EDIT:"
    )

    print(edited_2)

    # -----------------------------------------------------
    # Test OTHER FINDINGS insertion
    # -----------------------------------------------------

    insertion_edits = [

        {
            "target_section": None,
            "parent_section": None,
            "edited_content": (
                "Mild degenerative fraying of the superior "
                "labrum without a displaced tear."
            ),
            "action": "insert",
        },

        {
            "target_section": None,
            "parent_section": None,
            "edited_content": (
                "Moderate subacromial-subdeltoid bursitis."
            ),
            "action": "insert",
        },

        {
            "target_section": None,
            "parent_section": None,
            "edited_content": (
                "Mild thickening and edema within the "
                "rotator interval and inferior glenohumeral capsule."
            ),
            "action": "insert",
        },
    ]

    final = apply_section_edits(
        template=edited_2,
        edits=insertion_edits,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WITH OTHER FINDINGS:"
    )

    print(final)

