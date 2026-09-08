import pandas as pd

from template_parser import normalize_template, parse_template


# ---------------------------------------------------------
# Parent -> child relationships
# ---------------------------------------------------------
#
# These are structural relationships observed in the
# templates. We are NOT using this to generate medical
# findings. It is only used to understand report layout.
#

PARENT_SECTIONS = {
    "TENDONS": {
    "SUPRASPINATUS",
    "INFRASPINATUS",
    "SUBSCAPULARIS",
    "TERES MINOR",
    "BICEPS BRACHII",
    "BICEPS BRACHII, LONG HEAD",
    "LONG HEAD",
    },

    "LIGAMENTS": {
        "GLENOHUMERAL",
        "ANTERIOR CRUCIATE LIGAMENT",
        "POSTERIOR CRUCIATE LIGAMENT",
        "MEDIAL COLLATERAL LIGAMENT",
        "LATERAL COLLATERAL LIGAMENT COMPLEX",
    },

    "JOINTS": {
        "GLENOHUMERAL JOINT",
        "ACROMIOCLAVICULAR JOINT",
    },

    "MENISCI": {
        "MEDIAL MENISCUS",
        "LATERAL MENISCUS",
    },

    "BICEPS ANCHOR COMPLEX": {
        "LONG HEAD",
    },
}


def normalize_name(name):
    """
    Normalize a section name for comparison.

    Example:
        'Supraspinatus' -> 'SUPRASPINATUS'
    """
    if name is None:
        return ""

    return " ".join(str(name).upper().split())


def infer_hierarchy(sections):
    """
    Convert the flat list produced by parse_template()
    into a hierarchical structure.

    We preserve the original order.

    Each node has:

        name
        content
        children

    Example:

        TENDONS
            SUPRASPINATUS
            INFRASPINATUS
            SUBSCAPULARIS

    """

    root = []

    current_parent = None

    for section in sections:

        name = section["name"]
        content = section["content"]

        # Unlabelled content
        if name is None:
            root.append({
                "name": None,
                "content": content,
                "children": []
            })

            continue

        normalized_name = normalize_name(name)

        # -------------------------------------------------
        # Check whether this section itself is a parent
        # -------------------------------------------------

        if normalized_name in PARENT_SECTIONS:

            node = {
                "name": name,
                "content": content,
                "children": []
            }

            root.append(node)

            current_parent = node

            continue

        # -------------------------------------------------
        # Check whether current section belongs to the
        # current parent.
        # -------------------------------------------------

        if current_parent is not None:

            parent_name = normalize_name(current_parent["name"])

            children = PARENT_SECTIONS.get(parent_name, set())

            if normalized_name in children:

                current_parent["children"].append({
                    "name": name,
                    "content": content,
                    "children": []
                })

                continue

        # -------------------------------------------------
        # Otherwise this is a normal top-level section.
        # -------------------------------------------------

        node = {
            "name": name,
            "content": content,
            "children": []
        }

        root.append(node)

        # This section is not automatically a parent.
        current_parent = None

    return root


def build_template_structure(template):
    """
    Complete pipeline:

        raw template
             ↓
        parse_template()
             ↓
        infer_hierarchy()
             ↓
        structured template
    """

    parsed = parse_template(template)

    hierarchy = infer_hierarchy(
        parsed["findings"]
    )

    return {
        "findings": hierarchy,
        "impression": parsed["impression"],
        "raw_template": parsed["raw_template"]
    }


def print_tree(nodes, level=0):
    """
    Pretty-print the hierarchy.
    """

    for node in nodes:

        name = node["name"]

        if name is None:
            name = "[UNLABELLED]"

        indent = "    " * level

        print(f"{indent}├── {name}")

        # Print children recursively
        if node["children"]:
            print_tree(
                node["children"],
                level + 1
            )


def print_structure(structure, template_number=None):

    if template_number is not None:
        print()
        print("#" * 80)
        print(f"TEMPLATE {template_number}")
        print("#" * 80)

    print("\nFINDINGS HIERARCHY")
    print("=" * 60)

    print_tree(
        structure["findings"]
    )

    print("\nIMPRESSION")
    print("=" * 60)

    print(structure["impression"])


def get_all_section_names(nodes):
    """
    Return all section names recursively.

    Useful later when we build routing logic.
    """

    names = []

    for node in nodes:

        if node["name"] is not None:
            names.append(node["name"])

        if node["children"]:
            names.extend(
                get_all_section_names(node["children"])
            )

    return names


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    train = pd.read_csv("data/train.csv")

    print("TRAIN CASES:", len(train))

    # -----------------------------------------------------
    # Get unique templates
    # -----------------------------------------------------

    unique_templates = []
    seen = set()

    for template in train["template_content"]:

        normalized = normalize_template(template)

        if normalized not in seen:

            seen.add(normalized)
            unique_templates.append(normalized)

    print("UNIQUE TEMPLATES:", len(unique_templates))

    # -----------------------------------------------------
    # Inspect first 10 templates
    # -----------------------------------------------------

    for i, template in enumerate(
        unique_templates[:10],
        start=1
    ):

        structure = build_template_structure(
            template
        )

        print_structure(
            structure,
            template_number=i
        )