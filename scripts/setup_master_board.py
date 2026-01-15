#!/usr/bin/env python3
"""
PAROL6 - GitHub Project Board Automation Tool

Purpose
-------
Automates population of the PAROL6 GitHub Project Board by:

  • Interactive GitHub login (no tokens required)
  • Parsing issue templates from .github/ISSUE_TEMPLATE
  • Creating missing issues safely
  • Preventing duplicate issue creation
  • Adding issues to GitHub Projects v2 board
  • Recovering gracefully from UI/network failures

Design Principles
-----------------
  • Deterministic behavior
  • Defensive error handling
  • Stable selectors (no fragile UI labels)
  • Clear logging for audit / thesis traceability
  • Future extensibility toward API automation

Author: PAROL6 Team
Version: 2.0.0
"""

import time
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# =============================================================================
# ================================ CONFIG =====================================
# =============================================================================

REPO_OWNER = "Abdulkareem771"
REPO_NAME  = "PAROL6_URDF"
REPO_URL   = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"

PROJECT_NAME = "PAROL6"
PROJECT_URL  = "https://github.com/users/Abdulkareem771/projects/1"

ISSUE_TEMPLATE_DIR = Path(".github/ISSUE_TEMPLATE")

HEADLESS = False                     # Visible browser for reliability
NAV_TIMEOUT_MS = 60_000
UI_DELAY = 1.5                       # UI stabilization delay

SKIP_EXISTING_ISSUES = True
MAX_NAV_RETRIES = 3


# =============================================================================
# =============================== LOGGING =====================================
# =============================================================================

def log(msg: str) -> None:
    print(f"[PAROL6-AUTO] {msg}")


# =============================================================================
# ============================ TEMPLATE PARSING ===============================
# =============================================================================

def parse_issue_template(filepath: Path) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse GitHub issue template YAML frontmatter and body.
    """

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        log(f"❌ Failed reading template {filepath.name}: {e}")
        return None, None

    if not content.startswith("---"):
        log(f"⚠️  Template missing YAML header: {filepath.name}")
        return None, None

    parts = content.split("---", 2)
    if len(parts) < 3:
        log(f"⚠️  Invalid YAML structure: {filepath.name}")
        return None, None

    yaml_block = parts[1]
    body = parts[2].strip()

    meta: Dict[str, object] = {}

    for line in yaml_block.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        meta[key.strip()] = val.strip().strip('"').strip("'")

    # Extract labels array
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
        log("⚠️  No issue templates found.")
    return templates


# =============================================================================
# ============================ BROWSER HELPERS ================================
# =============================================================================

def safe_goto(page, url: str) -> bool:
    """
    Robust navigation with retry logic.
    """
    for attempt in range(1, MAX_NAV_RETRIES + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
            time.sleep(UI_DELAY)
            return True
        except PlaywrightTimeoutError:
            log(f"⚠️  Navigation timeout (attempt {attempt}/{MAX_NAV_RETRIES}) → {url}")
            time.sleep(2)

    log(f"❌ Navigation failed permanently → {url}")
    return False


def wait_for_login(page) -> None:
    log("🔵 Opening GitHub login page...")
    page.goto("https://github.com/login", wait_until="domcontentloaded")

    log("👇 ACTION REQUIRED:")
    log("   Please log in manually until GitHub dashboard loads.")
    log("   Automation will resume automatically.")

    page.wait_for_url("https://github.com/", timeout=0)
    log("✅ Login detected.")


# =============================================================================
# ============================ GITHUB OPERATIONS ==============================
# =============================================================================

def issue_exists(page, title: str) -> bool:
    """
    Detect if issue already exists by searching repository.
    """

    encoded_title = quote_plus(title)
    search_url = f"{REPO_URL}/issues?q=is%3Aissue+{encoded_title}"

    if not safe_goto(page, search_url):
        log("⚠️  Issue search failed — assuming issue does not exist.")
        return False

    try:
        matches = page.get_by_text(title, exact=False).count()
        return matches > 0
    except Exception:
        return False


def create_issue(page, title: str, body: str) -> bool:
    """
    Create new GitHub issue safely.
    """

    log(f"📝 Creating issue: {title}")

    if not safe_goto(page, f"{REPO_URL}/issues/new"):
        return False

    try:
        # Robust selectors (avoid fragile placeholders)
        page.locator("input[name='issue[title]']").fill(title)
        page.locator("textarea").first.fill(body)

        page.get_by_text("Submit new issue").click()
        page.wait_for_load_state("networkidle")

        return True

    except Exception as e:
        log(f"❌ Issue creation failed [{title}]: {e}")
        return False


def add_issue_to_project(page) -> bool:
    """
    Attempt to add current issue to GitHub Project board.
    """

    try:
        page.get_by_role("button", name="Projects").click()
        time.sleep(1)

        page.get_by_placeholder("Filter projects").fill(PROJECT_NAME)
        time.sleep(1)

        page.get_by_text(PROJECT_NAME, exact=False).first.click()
        page.keyboard.press("Escape")

        log(f"   📌 Linked to project: {PROJECT_NAME}")
        return True

    except Exception as e:
        log(f"⚠️  Project linking failed: {e}")
        return False


# =============================================================================
# ================================ MAIN =======================================
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

        # ---------------- Processing ----------------
        created = 0
        skipped = 0
        failed  = 0

        log("📂 Scanning templates...")

        for template_path in templates:

            if "bug" in template_path.name.lower():
                log(f"⏭️  Skipping template: {template_path.name}")
                continue

            meta, body = parse_issue_template(template_path)
            if not meta or not meta.get("title"):
                log(f"⚠️  Invalid template skipped: {template_path.name}")
                continue

            title = str(meta["title"])

            if SKIP_EXISTING_ISSUES and issue_exists(page, title):
                log(f"♻️  Already exists: {title}")
                skipped += 1
                continue

            if not create_issue(page, title, body):
                failed += 1
                continue

            time.sleep(1)

            add_issue_to_project(page)
            created += 1
            time.sleep(2)

        # ---------------- Summary ----------------
        log("========================================")
        log("🎉 Automation Completed")
        log(f"   Created : {created}")
        log(f"   Skipped : {skipped}")
        log(f"   Failed  : {failed}")
        log("========================================")

        input("Press ENTER to close browser...")
        browser.close()


# =============================================================================
# ============================== ENTRYPOINT ==================================
# =============================================================================

if __name__ == "__main__":
    run_automation()
