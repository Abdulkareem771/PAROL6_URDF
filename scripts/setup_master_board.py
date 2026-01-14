#!/usr/bin/env python3
"""
PAROL6 - GitHub Project Board Automation

Purpose:
---------
Automates population of the PAROL6 GitHub Project Board by:
  - Logging into GitHub interactively
  - Opening the Project Board (Projects v2 compatible)
  - Parsing issue templates from .github/ISSUE_TEMPLATE
  - Creating issues automatically
  - Adding issues to the Project Board
  - Avoiding duplicate issue creation

This script is intentionally conservative and UI-based
to avoid GitHub API token management and permission complexity.

Designed for:
  - Repeatable onboarding
  - Thesis evidence reproducibility
  - Long-term maintainability

Author: PAROL6 Team
Version: 1.1.0
"""

import time
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# =============================================================================
# ============================= CONFIGURATION =================================
# =============================================================================

# Repository metadata
REPO_OWNER = "Abdulkareem771"
REPO_NAME = "PAROL6_URDF"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"

# GitHub Project (Projects v2 URL — MUST be direct link)
PROJECT_NAME = "PAROL6"
PROJECT_URL = "https://github.com/users/Abdulkareem771/projects/1"

# Local templates path
ISSUE_TEMPLATE_DIR = Path(".github/ISSUE_TEMPLATE")

# Browser behavior
HEADLESS = False                 # Keep visible for debugging
NAVIGATION_TIMEOUT_MS = 60_000
UI_STABILIZATION_DELAY = 2.0     # seconds

# Safety
SKIP_EXISTING_ISSUES = True      # Prevent duplicates


# =============================================================================
# ============================= UTILITIES =====================================
# =============================================================================

def log(msg: str) -> None:
    print(f"[PAROL6-AUTO] {msg}")


def parse_issue_template(filepath: Path) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Extracts metadata and body from GitHub issue template markdown.

    Supports YAML frontmatter:
    ---
    title: "My Title"
    labels: ["vision", "hardware"]
    ---
    Body text...
    """

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        log(f"❌ Failed reading template {filepath}: {e}")
        return None, None

    if not content.startswith("---"):
        log(f"⚠️  Template has no YAML header: {filepath.name}")
        return None, None

    parts = content.split("---", 2)
    if len(parts) < 3:
        log(f"⚠️  Invalid YAML block in {filepath.name}")
        return None, None

    yaml_block = parts[1]
    body = parts[2].strip()

    meta = {}
    for line in yaml_block.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        meta[key.strip()] = val.strip().strip('"').strip("'")

    # Extract labels list if present
    labels_match = re.search(r"labels:\s*\[(.*?)\]", yaml_block)
    if labels_match:
        labels = [
            l.strip().strip('"').strip("'")
            for l in labels_match.group(1).split(",")
            if l.strip()
        ]
        meta["labels"] = labels
    else:
        meta["labels"] = []

    return meta, body


def collect_templates() -> List[Path]:
    if not ISSUE_TEMPLATE_DIR.exists():
        log("❌ ISSUE_TEMPLATE directory not found.")
        return []

    templates = sorted(ISSUE_TEMPLATE_DIR.glob("*.md"))
    if not templates:
        log("⚠️  No templates found.")
    return templates


# =============================================================================
# ============================ PLAYWRIGHT HELPERS =============================
# =============================================================================

def safe_goto(page, url: str) -> bool:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        time.sleep(UI_STABILIZATION_DELAY)
        return True
    except PlaywrightTimeoutError:
        log(f"❌ Timeout navigating to: {url}")
        return False


def wait_for_login(page) -> None:
    log("🔵 Opening GitHub login page...")
    page.goto("https://github.com/login", wait_until="domcontentloaded")

    log("👇 ACTION REQUIRED:")
    log("   Please log in manually until you reach your GitHub dashboard.")
    log("   The script will continue automatically.")

    # Wait until GitHub home loads after login
    page.wait_for_url("https://github.com/", timeout=0)
    log("✅ Login detected.")


def issue_exists(page, title: str) -> bool:
    """
    Checks if an issue with identical title already exists in repo.
    """
    search_url = f"{REPO_URL}/issues?q=is%3Aissue+{title.replace(' ', '+')}"
    if not safe_goto(page, search_url):
        return False

    try:
        if page.get_by_text(title, exact=False).count() > 0:
            return True
    except Exception:
        pass

    return False


def create_issue(page, title: str, body: str) -> bool:
    """
    Creates a new GitHub issue.
    """
    log(f"📝 Creating issue: {title}")

    if not safe_goto(page, f"{REPO_URL}/issues/new"):
        return False

    try:
        page.get_by_placeholder("Title").fill(title)
        page.get_by_placeholder("Leave a comment").fill(body)
        page.get_by_text("Submit new issue").click()
        page.wait_for_load_state("networkidle")
        return True
    except Exception as e:
        log(f"❌ Failed creating issue '{title}': {e}")
        return False


def add_issue_to_project(page) -> bool:
    """
    Adds currently open issue to the project board.
    """
    try:
        page.get_by_role("button", name="Projects").click()
        time.sleep(1)

        page.get_by_placeholder("Filter projects").fill(PROJECT_NAME)
        time.sleep(1)

        page.get_by_text(PROJECT_NAME, exact=False).first.click()
        page.keyboard.press("Escape")

        return True
    except Exception as e:
        log(f"⚠️  Could not auto-add issue to project: {e}")
        return False


# =============================================================================
# ================================ MAIN LOGIC =================================
# =============================================================================

def run_automation() -> None:
    log("🚀 Starting PAROL6 Project Board Automation")

    templates = collect_templates()
    if not templates:
        log("❌ No templates found — exiting.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context()
        page = context.new_page()

        # ---------------- Login ----------------
        wait_for_login(page)

        # ---------------- Open Project ----------------
        log("🔵 Opening Project Board...")
        if not safe_goto(page, PROJECT_URL):
            log("❌ Failed to open Project Board. Aborting.")
            return

        log("✅ Project Board opened successfully.")

        # ---------------- Create Issues ----------------
        created_count = 0
        skipped_count = 0

        log("📂 Scanning templates...")

        for template_path in templates:
            if "bug" in template_path.name.lower():
                log(f"⏭️  Skipping template: {template_path.name}")
                continue

            meta, body = parse_issue_template(template_path)
            if not meta or not meta.get("title"):
                log(f"⚠️  Invalid template skipped: {template_path.name}")
                continue

            title = meta["title"]

            if SKIP_EXISTING_ISSUES and issue_exists(page, title):
                log(f"♻️  Already exists: {title}")
                skipped_count += 1
                continue

            if not create_issue(page, title, body):
                continue

            time.sleep(1)

            if add_issue_to_project(page):
                log("   ✅ Added to project")
            else:
                log("   ⚠️  Please add manually if missing")

            created_count += 1
            time.sleep(2)  # polite pacing

        # ---------------- Summary ----------------
        log("========================================")
        log("🎉 Automation Completed")
        log(f"   Created Issues : {created_count}")
        log(f"   Skipped Issues : {skipped_count}")
        log("👉 Review the Project Board for correctness.")
        log("========================================")

        input("Press ENTER to close browser...")
        browser.close()


# =============================================================================
# ================================ ENTRYPOINT =================================
# =============================================================================

if __name__ == "__main__":
    run_automation()
