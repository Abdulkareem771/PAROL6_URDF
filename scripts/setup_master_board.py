#!/usr/bin/env python3
"""
PAROL6 Project Board Automation (V4)

Author: PAROL6 Project
Purpose:
    Fully automates GitHub Project V2 taxonomy:
        - Creates required fields if missing
        - Creates select options if missing
        - Automatically classifies issues
        - Idempotent and safe to re-run
        - Friendly for future task additions

Dependencies:
    - gh CLI authenticated
    - jq installed

Run:
    ./scripts/setup_master_board.py
"""

import json
import subprocess
from typing import Dict, List

# ============================================================
# ======================= CONFIGURATION ======================
# ============================================================

OWNER = "Abdulkareem771"
REPO  = "PAROL6_URDF"
PROJECT_NUMBER = 1     # https://github.com/users/<user>/projects/1

# ---------------- Field Definitions ----------------

FIELDS = {
    "Domain": [
        "hardware",
        "esp32",
        "control",
        "foc",
        "ros",
        "vision",
        "robotics",
        "ai",
        "other",
    ],
    "Layer": [
        "hardware",
        "firmware",
        "middleware",
        "control",
        "perception",
        "planning",
        "integration",
        "tooling",
        "documentation",
    ],
    "Maturity": [
        "idea",
        "prototype",
        "validated",
        "integrated",
        "production",
        "deprecated",
    ],
}

DEFAULT_MATURITY = "idea"

# ---------------- Classification Rules ----------------

DOMAIN_KEYWORDS = {
    "esp32":    ["esp32", "uart", "serial", "firmware"],
    "foc":      ["foc", "current", "torque"],
    "control":  ["control", "pid", "trajectory"],
    "ros":      ["ros", "node", "launch"],
    "vision":   ["vision", "camera", "yolo", "marker"],
    "robotics": ["robot", "kinematic", "moveit"],
    "ai":       ["ai", "model", "dataset", "training"],
    "hardware": ["driver", "motor", "pcb", "sensor"],
}

DOMAIN_TO_LAYER = {
    "hardware": "hardware",
    "esp32": "firmware",
    "ros": "middleware",
    "control": "control",
    "foc": "control",
    "vision": "perception",
    "robotics": "planning",
    "ai": "perception",
    "other": "tooling",
}

# ============================================================
# ====================== UTILITIES ===========================
# ============================================================

def run(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def gql(query: str) -> dict:
    out = run(["gh", "api", "graphql", "-f", f"query={query}"])
    return json.loads(out)


def log(msg: str):
    print(f"[AUTO] {msg}")

# ============================================================
# ====================== DISCOVERY ===========================
# ============================================================

def get_project_id() -> str:
    query = f"""
    query {{
      user(login: "{OWNER}") {{
        projectV2(number: {PROJECT_NUMBER}) {{
          id
        }}
      }}
    }}
    """
    data = gql(query)
    return data["data"]["user"]["projectV2"]["id"]


def get_fields(project_id: str) -> Dict[str, dict]:
    query = f"""
    query {{
      node(id: "{project_id}") {{
        ... on ProjectV2 {{
          fields(first: 50) {{
            nodes {{
              ... on ProjectV2FieldCommon {{
                id
                name
                dataType
              }}
              ... on ProjectV2SingleSelectField {{
                options {{
                  id
                  name
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    data = gql(query)
    fields = {}
    for f in data["data"]["node"]["fields"]["nodes"]:
        fields[f["name"]] = f
    return fields


def get_project_items(project_id: str) -> List[dict]:
    query = f"""
    query {{
      node(id: "{project_id}") {{
        ... on ProjectV2 {{
          items(first: 100) {{
            nodes {{
              id
              content {{
                ... on Issue {{
                  id
                  number
                  title
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    data = gql(query)
    return data["data"]["node"]["items"]["nodes"]

# ============================================================
# ====================== FIELD CONTROL =======================
# ============================================================

def create_field(project_id: str, name: str, options: List[str]):
    # Construct options string for GraphQL: [{name: "Opt1", color: GRAY, description: ""}, ...]
    # We use GRAY as default color.
    opts_str = ", ".join([f'{{name: "{o}", color: GRAY, description: ""}}' for o in options])
    
    mutation = f"""
    mutation {{
      createProjectV2Field(input: {{
        projectId: "{project_id}",
        name: "{name}",
        dataType: SINGLE_SELECT,
        singleSelectOptions: [{opts_str}]
      }}) {{
        projectV2Field {{
          ... on ProjectV2FieldCommon {{
            id
          }}
        }}
      }}
    }}
    """
    gql(mutation)
    log(f"Field created: {name}")


def add_option(field_id: str, option: str):
    # Depending on API, add_option might also require color/desc if strict.
    # createProjectV2FieldOption input: name, fieldId, (color, description optional?)
    # Usually they are optional for append, but let's see. 
    # To be safe, adding them explicitly is better.
    mutation = f"""
    mutation {{
      createProjectV2FieldOption(input: {{
        fieldId: "{field_id}",
        name: "{option}",
        color: GRAY,
        description: ""
      }}) {{
        projectV2FieldOption {{ id }}
      }}
    }}
    """
    gql(mutation)
    log(f"  Option added: {option}")


def ensure_fields(project_id: str) -> Dict[str, dict]:
    fields = get_fields(project_id)

    for field_name, options_list in FIELDS.items():
        if field_name not in fields:
            create_field(project_id, field_name, options_list)
            # Fetch again to get IDs
            fields = get_fields(project_id)

        field = fields[field_name]
        existing = {o["name"] for o in field.get("options", [])}

        for opt in options_list:
            if opt not in existing:
                add_option(field["id"], opt)

    return get_fields(project_id)

# ============================================================
# ====================== CLASSIFICATION ======================
# ============================================================

def classify_domain(title: str) -> str:
    t = title.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in t for k in keywords):
            return domain
    return "other"


def classify_layer(domain: str) -> str:
    return DOMAIN_TO_LAYER.get(domain, "tooling")

# ============================================================
# ====================== FIELD UPDATE ========================
# ============================================================

def set_field(item_id: str, field_id: str, option_id: str):
    mutation = f"""
    mutation {{
      updateProjectV2ItemFieldValue(input: {{
        projectId: "{PROJECT_ID}",
        itemId: "{item_id}",
        fieldId: "{field_id}",
        value: {{ singleSelectOptionId: "{option_id}" }}
      }}) {{
        projectV2Item {{ id }}
      }}
    }}
    """
    gql(mutation)


def get_option_id(field: dict, name: str) -> str:
    for opt in field["options"]:
        if opt["name"] == name:
            return opt["id"]
    # If not found, create it on the fly? Or fail.
    # Should be created by ensure_fields.
    raise RuntimeError(f"Option not found: {name}")

# ============================================================
# ========================== MAIN ============================
# ============================================================

PROJECT_ID = ""

def main():
    global PROJECT_ID
    log("Starting Project Automation")

    PROJECT_ID = get_project_id()
    fields = ensure_fields(PROJECT_ID)
    items  = get_project_items(PROJECT_ID)

    domain_field   = fields["Domain"]
    layer_field    = fields["Layer"]
    maturity_field = fields["Maturity"]

    for item in items:
        if not item or not item.get("content"):
            continue

        title = item["content"].get("title", "Unknown")
        item_id = item["id"]
        number = item["content"].get("number", "?")

        domain = classify_domain(title)
        layer  = classify_layer(domain)
        
        log(f"Processing #{number} | {title}")
        log(f"  → Domain   = {domain}")
        log(f"  → Layer    = {layer}")
        log(f"  → Maturity = {DEFAULT_MATURITY}")

        try:
            set_field(
                item_id,
                domain_field["id"],
                get_option_id(domain_field, domain),
            )

            set_field(
                item_id,
                layer_field["id"],
                get_option_id(layer_field, layer),
            )
            
            set_field(
                item_id,
                maturity_field["id"],
                get_option_id(maturity_field, DEFAULT_MATURITY),
            )
        except Exception as e:
            log(f"  ❌ Failed to update item: {e}")

    log("Automation completed successfully")

if __name__ == "__main__":
    main()
