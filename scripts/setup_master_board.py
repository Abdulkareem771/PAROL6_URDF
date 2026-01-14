import time
import os
import glob
import re
from playwright.sync_api import sync_playwright

REPO_URL = "https://github.com/Abdulkareem771/PAROL6_URDF"
PROJECT_NAME = "PAROL6"

def parse_issue_template(filepath):
    """Extracts title, body, and labels from markdown template."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse generic YAML frontmatter
    meta = {}
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_block = parts[1]
            body = parts[2].strip()
            
            for line in yaml_block.split('\n'):
                if ':' in line:
                    key, val = line.split(':', 1)
                    meta[key.strip()] = val.strip().strip('"').strip("'")
            
            # Extract labels list specifically
            labels_match = re.search(r'labels: \[(.*?)\]', yaml_block)
            if labels_match:
                meta['labels'] = [l.strip().strip('"').strip("'") for l in labels_match.group(1).split(',')]
            
            return meta, body
    return None, None

def get_track_from_labels(labels):
    """Maps labels to track names."""
    if not labels: return None
    if 'vision' in labels: return '👁️ Vision'
    if 'ai' in labels: return '🧠 AI'
    if 'mobile' in labels: return '📱 Mobile'
    if 'hardware' in labels or 'day2' in labels: return '🤖 Hardware'
    return None

def run_automation():
    print(f"🚀 Starting automation for {PROJECT_NAME}...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # 1. Login Logic
        print("🔵 Navigating to GitHub Login...")
        page.goto("https://github.com/login")
        print("👇 ACTION REQUIRED: Please log in repeatedly until you reach your dashboard.")
        try:
            page.wait_for_url("https://github.com/", timeout=0)
            print("✅ Login Success!")
        except:
            print("❌ Login timed out.")
            return

        # 2. Go to Project
        print(f"🔵 Opening Project: {PROJECT_NAME}")
        projects_url = f"{REPO_URL}/projects?query=is%3Aopen"
        page.goto(projects_url)
        
        # Click the project link (finding by partial text)
        try:
            page.get_by_text(PROJECT_NAME, exact=False).first.click()
            page.wait_for_load_state('networkidle')
            print("✅ Project Board Opened")
        except Exception as e:
            print(f"❌ Could not find project '{PROJECT_NAME}'. Please open it manually.")
            input("Press Enter once Project Board is open...")

        # 3. Create 'Track' Field (User said they did manual work, but let's verify/add)
        print("\n🔧 Configuring 'Track' Field...")
        # Note: UI automation for creating fields is fragile due to dynamic IDs.
        # We will skip direct field creation to avoid breaking and focus on Issue Creation which is high value.
        print("ℹ️  Skipping field creation (assumed manual). Focusing on Issue Population.")

        # 4. Create Issues from Templates
        print("\n📝 Creating Issues from Templates...")
        templates = glob.glob(".github/ISSUE_TEMPLATE/*.md")
        
        for template_path in sorted(templates):
            # Skip templates that shouldn't be auto-created (like generic bug_report) if desired, 
            # but user wants to populate the board, so let's do the specific phase ones.
            if "bug_report" in template_path: continue
            
            meta, body = parse_issue_template(template_path)
            if not meta or not meta.get('title'): continue
            
            title = meta['title']
            print(f"   Creating: {title}")
            
            # Go to New Issue page
            page.goto(f"{REPO_URL}/issues/new")
            
            # If template chooser appears, skip it by going directly or clicking "Open a blank issue"
            # Actually, easiest is just filling the blank issue form
            page.goto(f"{REPO_URL}/issues/new/choose")
            
            # Check if this exact template exists in list and click it?
            # Creating from scratch is safer to ensure title/body match EXACTLY what's on disk
            page.goto(f"{REPO_URL}/issues/new")
            
            # Fill Title
            page.get_by_placeholder("Title").fill(title)
            
            # Fill Body (using code view usually safer but plain text area works)
            page.get_by_placeholder("Leave a comment").fill(body)
            
            # Add to Project (Metadata Sidebar)
            # This is complex in UI. Easier to just create the issue first.
            
            # Submit
            page.get_by_text("Submit new issue").click()
            page.wait_for_load_state('networkidle')
            print("     ✅ Issue Created")
            
            # Now Add to Project (Post-creation)
            # Find the "Projects" gear on the right sidebar
            try:
                page.get_by_role("button", name="Projects").click()
                page.get_by_placeholder("Filter projects").fill(PROJECT_NAME)
                # Click the project in the menu
                page.get_by_text(PROJECT_NAME, exact=False).first.click()
                # Click away to save
                page.keyboard.press("Escape")
                print("     ✅ Added to Project")
            except:
                print("     ⚠️ Failed to add to Project (do it manually)")
                
            time.sleep(1) # Rate limit kindness

        print("\n✅ Automation Complete!")
        print("👉 Go to your Project Board and use the 'Track' field to organize items.")
        input("Press Enter to close browser...")
        browser.close()

if __name__ == "__main__":
    run_automation()
