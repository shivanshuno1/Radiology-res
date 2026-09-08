#src/routing.py
import pandas as pd

from template_parser import normalize_template
from template_structure import build_template_structure


# =========================================================
# ROUTING DATA STRUCTURE
# =========================================================

def create_route(
    finding,
    section,
    parent=None,
    action="modify"
):
    """
    Create a routing decision.

    Example:

    {
        "finding": "Partial thickness supraspinatus tear",
        "section": "SUPRASPINATUS",
        "parent": "TENDONS",
        "action": "modify"
    }
    """

    return {
        "finding": finding,
        "section": section,
        "parent": parent,
        "action": action
    }


# =========================================================
# SECTION SEARCH
# =========================================================

def find_section(nodes, target_name, parent=None):
    """
    Recursively search the hierarchical template.

    Returns the matching section node and its parent.

    Example:

        find_section(
            structure["findings"],
            "SUPRASPINATUS"
        )
    """

    target = target_name.strip().upper()

    for node in nodes:

        name = node["name"]

        if name is not None:

            if name.strip().upper() == target:
                return node, parent

        # Search children
        if node["children"]:

            result = find_section(
                node["children"],
                target_name,
                parent=node
            )

            if result is not None:
                return result

    return None


# =========================================================
# GET ALL SECTIONS
# =========================================================

def get_sections(nodes, parent=None):
    """
    Flatten the hierarchy while preserving parent information.

    Returns:

        [
            {
                "name": "SUPRASPINATUS",
                "parent": "TENDONS"
            },
            ...
        ]
    """

    sections = []

    for node in nodes:

        name = node["name"]

        if name is not None:

            sections.append({
                "name": name,
                "parent": (
                    parent["name"]
                    if parent is not None
                    else None
                )
            })

        if node["children"]:

            sections.extend(
                get_sections(
                    node["children"],
                    parent=node
                )
            )

    return sections


# =========================================================
# FIND POSSIBLE SECTION NAMES
# =========================================================

def get_section_names(structure):
    """
    Return every labelled section in a template.
    """

    return [
        section["name"]
        for section in get_sections(
            structure["findings"]
        )
    ]


# =========================================================
# SIMPLE KEYWORD ROUTER
# =========================================================
#
# IMPORTANT:
# This is NOT the final medical routing system.
#
# It is only a deterministic baseline so that we can test
# the architecture before introducing Gemini.
#
# Later Gemini will perform the semantic extraction/routing.
# =========================================================

ROUTING_KEYWORDS = {

    # -----------------------------
    # Shoulder MRI
    # -----------------------------

    "SUPRASPINATUS": [
        "supraspinatus"
    ],

    "INFRASPINATUS": [
        "infraspinatus"
    ],

    "SUBSCAPULARIS": [
        "subscapularis"
    ],

    "TERES MINOR": [
        "teres minor"
    ],

    "BICEPS BRACHII, LONG HEAD": [
        "biceps",
        "long head of the biceps",
        "biceps tendon"
    ],

    "GLENOHUMERAL JOINT": [
        "glenohumeral joint",
        "glenohumeral effusion",
        "glenohumeral"
    ],

    "ACROMIOCLAVICULAR JOINT": [
        "acromioclavicular",
        "acromioclavicular joint",
        "ac joint"
    ],

    # -----------------------------
    # Knee MRI
    # -----------------------------

    "MEDIAL MENISCUS": [
        "medial meniscus"
    ],

    "LATERAL MENISCUS": [
        "lateral meniscus"
    ],

    "ANTERIOR CRUCIATE LIGAMENT": [
        "anterior cruciate ligament",
        "acl"
    ],

    "POSTERIOR CRUCIATE LIGAMENT": [
        "posterior cruciate ligament",
        "pcl"
    ],

    "MEDIAL COLLATERAL LIGAMENT": [
        "medial collateral ligament",
        "mcl"
    ],

    "LATERAL COLLATERAL LIGAMENT COMPLEX": [
        "lateral collateral ligament",
        "lcl"
    ],

    # -----------------------------
    # Chest X-ray
    # -----------------------------

    "LUNGS": [
        "lung",
        "lungs",
        "opacity",
        "opacities",
        "infiltrate",
        "consolidation",
        "atelectasis",
        "pneumonia"
    ],

    "PLEURAL SPACES": [
        "pleural effusion",
        "pleural effusions",
        "pneumothorax",
        "pleural"
    ],

    "HEART": [
        "heart",
        "cardiomegaly"
    ],

    "MEDIASTINUM/HILA": [
        "mediastinum",
        "hila",
        "hilar"
    ],

    "DIAPHRAGM": [
        "diaphragm"
    ],

    # -----------------------------
    # Spine
    # -----------------------------

    "VERTEBRAE": [
        "vertebra",
        "vertebrae",
        "vertebral",
        "fracture",
        "scoliosis",
        "listhesis",
        "retrolisthesis",
        "anterolisthesis",
        "osteophyte",
        "osteophytes"
    ],

    "DISC SPACES": [
        "disc space",
        "disc spaces",
        "disc narrowing",
        "disc height",
        "degenerative disc"
    ],

    "DISCS/DEGENERATIVE CHANGES": [
        "disc",
        "disc bulge",
        "disc protrusion",
        "disc herniation",
        "degenerative changes",
        "degenerative disc disease"
    ],

    "SPINAL CORD": [
        "spinal cord",
        "cord"
    ],

    "PARASPINAL SOFT TISSUES": [
        "paraspinal",
        "paravertebral"
    ],

    # -----------------------------
    # Brain
    # -----------------------------

    "BRAIN": [
        "brain",
        "infarct",
        "hemorrhage",
        "hemorrhage",
        "mass",
        "ischemia",
        "white matter"
    ],

    "VENTRICLES": [
        "ventricle",
        "ventricles",
        "ventricular"
    ],

    "ORBITS": [
        "orbit",
        "orbits"
    ],

    "SINUSES AND MASTOIDS": [
        "sinus",
        "sinuses",
        "mastoid",
        "mastoiditis"
    ],

    # -----------------------------
    # Generic sections
    # -----------------------------

    "BONES": [
        "bone",
        "bones",
        "fracture",
        "osseous",
        "osteophyte",
        "osteophytes",
        "marrow"
    ],

    "SOFT TISSUES": [
        "soft tissue",
        "soft tissues",
        "subcutaneous"
    ],

    "OTHER FINDINGS": [
        "visualized",
        "incidental"
    ]
}


# =========================================================
# KEYWORD ROUTING
# =========================================================

def keyword_route_finding(
    finding,
    structure
):
    """
    Try to route a finding using deterministic keywords.

    Returns:

        route dictionary

    or

        None

    if no suitable section is found.
    """

    finding_lower = finding.lower()

    available_sections = {
        section["name"].upper(): section
        for section in get_sections(
            structure["findings"]
        )
    }

    candidates = []

    for section_name, keywords in ROUTING_KEYWORDS.items():

        # Section does not exist in this template
        if section_name not in available_sections:
            continue

        for keyword in keywords:

            if keyword.lower() in finding_lower:

                candidates.append(
                    (
                        len(keyword),
                        section_name
                    )
                )

    if not candidates:
        return None

    # Prefer the most specific/longest matching phrase
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_section = candidates[0][1]

    section_node, parent = find_section(
        structure["findings"],
        best_section
    )

    return create_route(
        finding=finding,
        section=section_node["name"],
        parent=(
            parent["name"]
            if parent is not None
            else None
        )
    )


# =========================================================
# ROUTE MULTIPLE FINDINGS
# =========================================================

def route_findings(
    findings,
    structure
):
    """
    Route a list of findings.

    Example:

        findings = [
            "Partial thickness supraspinatus tear",
            "Moderate glenohumeral effusion"
        ]

    Returns:

        routes
        unrouted
    """

    routes = []
    unrouted = []

    for finding in findings:

        route = keyword_route_finding(
            finding,
            structure
        )

        if route is not None:
            routes.append(route)
        else:
            unrouted.append(finding)

    return {
        "routes": routes,
        "unrouted": unrouted
    }


# =========================================================
# DISPLAY ROUTING
# =========================================================

def print_routes(result):

    print("\nROUTING RESULTS")
    print("=" * 60)

    print("\nROUTED FINDINGS:")

    for route in result["routes"]:

        parent = route["parent"]

        if parent:
            location = f"{parent} -> {route['section']}"
        else:
            location = route["section"]

        print(f"\nFinding : {route['finding']}")
        print(f"Route   : {location}")
        print(f"Action  : {route['action']}")

    print("\nUNROUTED FINDINGS:")

    if not result["unrouted"]:
        print("None")

    else:

        for finding in result["unrouted"]:
            print(f"- {finding}")


# =========================================================
# TEST WITH REAL TRAIN DATA
# =========================================================

if __name__ == "__main__":

    train = pd.read_csv(
        "data/train.csv"
    )

    print("TRAIN CASES:", len(train))

    # -----------------------------------------------------
    # Pick a shoulder MRI case containing a supraspinatus
    # finding if possible.
    # -----------------------------------------------------

    selected_case = None

    for _, row in train.iterrows():

        dictation = str(row["dictation"])

        if (
            "supraspinatus" in dictation.lower()
            and "glenohumeral" in dictation.lower()
        ):

            selected_case = row
            break

    if selected_case is None:

        print(
            "\nNo suitable shoulder case found."
        )

    else:

        print("\nSELECTED CASE")
        print("=" * 60)

        print(
            "Case ID:",
            selected_case["case_id"]
        )

        print(
            "\nDICTATION:"
        )

        print(
            selected_case["dictation"]
        )

        # -------------------------------------------------
        # Build template hierarchy
        # -------------------------------------------------

        structure = build_template_structure(
            selected_case["template_content"]
        )

        # -------------------------------------------------
        # Example findings for testing
        #
        # We deliberately provide individual findings
        # rather than the complete dictation.
        # -------------------------------------------------

        findings = [
            "Partial thickness tear of the supraspinatus tendon.",
            "Moderate glenohumeral joint effusion."
        ]

        # -------------------------------------------------
        # Route findings
        # -------------------------------------------------

        result = route_findings(
            findings,
            structure
        )

        print_routes(result)