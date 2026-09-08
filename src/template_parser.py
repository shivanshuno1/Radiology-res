import pandas as pd
import re


# ---------------------------------------------------------
# 1. NORMALIZE TEMPLATE
# ---------------------------------------------------------

def normalize_template(template):

    if pd.isna(template):
        return ""

    template = str(template)

    template = template.replace("\\n", "\n")
    template = template.replace("\r\n", "\n")
    template = template.replace("\r", "\n")

    lines = [line.rstrip() for line in template.splitlines()]

    cleaned = []
    previous_blank = False

    for line in lines:

        if line.strip() == "":

            if not previous_blank:
                cleaned.append("")

            previous_blank = True

        else:

            cleaned.append(line)
            previous_blank = False

    return "\n".join(cleaned).strip()


# ---------------------------------------------------------
# 2. CHECK FOR "NAME: CONTENT"
# ---------------------------------------------------------

def split_inline_section(line):
    """
    Detect:

        BONES: No acute fracture.
        JOINTS: Normal.

    Returns:

        ("BONES", "No acute fracture.")

    Otherwise returns:

        (None, None)
    """

    stripped = line.strip()

    if not stripped:
        return None, None

    # Find first colon
    if ":" not in stripped:
        return None, None

    name, content = stripped.split(":", 1)

    name = name.strip()
    content = content.strip()

    if not name:
        return None, None

    # Avoid treating ordinary sentences containing ":" as sections
    if not re.match(
        r"^[A-Za-z0-9][A-Za-z0-9 /&(),.'\-]*$",
        name
    ):
        return None, None

    # FINDINGS / IMPRESSION handled separately
    if name.upper() in ["FINDINGS", "IMPRESSION"]:
        return None, None

    return name, content


# ---------------------------------------------------------
# 3. STANDALONE SECTION HEADER
# ---------------------------------------------------------

def is_standalone_header(line):

    stripped = line.strip()

    if not stripped:
        return False

    if not stripped.endswith(":"):
        return False

    name = stripped[:-1].strip()

    if not name:
        return False

    if name.upper() in ["FINDINGS", "IMPRESSION"]:
        return False

    if not re.match(
        r"^[A-Za-z0-9][A-Za-z0-9 /&(),.'\-]*$",
        name
    ):
        return False

    return True


# ---------------------------------------------------------
# 4. PARSE TEMPLATE
# ---------------------------------------------------------
def parse_template(template):

    template = normalize_template(template)

    lines = template.split("\n")

    structure = {
        "findings": [],
        "impression": "",
        "raw_template": template
    }

    current_section = None
    current_content = []

    inside_impression = False

    def save_section():

        nonlocal current_section
        nonlocal current_content

        if current_section is None:
            return

        content = "\n".join(current_content).strip()

        structure["findings"].append({
            "name": current_section,
            "content": content
        })

        current_section = None
        current_content = []

    for line in lines:

        stripped = line.strip()

        # -------------------------------------------------
        # FINDINGS
        # -------------------------------------------------

        if stripped.upper() == "FINDINGS:":
            inside_impression = False
            continue

        # -------------------------------------------------
        # IMPRESSION
        # -------------------------------------------------

        if stripped.upper() == "IMPRESSION:":

            save_section()

            inside_impression = True
            continue

        # -------------------------------------------------
        # IMPRESSION CONTENT
        # -------------------------------------------------

        if inside_impression:

            structure["impression"] += line + "\n"
            continue

        # -------------------------------------------------
        # CASE 1:
        #
        # BONES: No fracture.
        #
        # -------------------------------------------------

        section_name, inline_content = split_inline_section(line)

        if section_name is not None:

            save_section()

            current_section = section_name

            if inline_content:
                current_content.append(inline_content)

            continue

        # -------------------------------------------------
        # CASE 2:
        #
        # BONES:
        # No fracture.
        #
        # -------------------------------------------------

        if is_standalone_header(line):

            save_section()

            current_section = stripped[:-1].strip()

            continue

        # -------------------------------------------------
        # NORMAL CONTENT
        # -------------------------------------------------

        if current_section is not None:

            # Blank line means the current labelled
            # section has ended.
            if stripped == "":

                save_section()

            else:

                current_content.append(line)

        else:

            # Keep unlabelled text
            if stripped:

                structure["findings"].append({
                    "name": None,
                    "content": stripped
                })

    # -------------------------------------------------
    # SAVE LAST SECTION
    # -------------------------------------------------

    save_section()

    structure["impression"] = structure["impression"].strip()

    return structure
# ---------------------------------------------------------
# 5. PRINT STRUCTURE
# ---------------------------------------------------------

def print_structure(structure):

    print("\nFINDINGS STRUCTURE")
    print("=" * 60)

    for section in structure["findings"]:

        name = section["name"]

        if name is None:

            print("  [UNLABELLED]")
            print(f"      {section['content']}")

        else:

            print(f"  ├── {name}")

            content = section["content"]

            if content:

                preview = content.replace("\n", " ")

                if len(preview) > 120:
                    preview = preview[:120] + "..."

                print(f"  │     {preview}")

    print("\nIMPRESSION")
    print("=" * 60)

    print(structure["impression"])


# ---------------------------------------------------------
# 6. TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    train = pd.read_csv("data/train.csv")

    print("TRAIN CASES:", len(train))

    unique_templates = []
    seen = set()

    for template in train["template_content"]:

        normalized = normalize_template(template)

        if normalized not in seen:

            seen.add(normalized)
            unique_templates.append(normalized)

    print("UNIQUE TEMPLATES:", len(unique_templates))

    # Print first 10
    for i, template in enumerate(unique_templates[:10], start=1):

        print("\n")
        print("#" * 80)
        print(f"TEMPLATE {i}")
        print("#" * 80)

        structure = parse_template(template)

        print_structure(structure)