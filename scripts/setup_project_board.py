import time
from playwright.sync_api import sync_playwright
import getpass

def setup_project_board():
    repo_url = input("Enter Repository URL (e.g., https://github.com/Abdulkareem771/PAROL6_URDF): ")
    if not repo_url:
        repo_url = "https://github.com/Abdulkareem771/PAROL6_URDF"
    
    print(f"Target Repository: {repo_url}")
    print("Launching browser... Please login when prompted.")

    with sync_playwright() as p:
        # Launch browser in headful mode so user can see and login
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Go to GitHub login
        page.goto("https://github.com/login")
        
        # Wait for user to login (check for avatar or dashboard)
        print("🔴 ACTION REQUIRED: Please log in to GitHub in the browser window.")
        print("Waiting for login to complete...")
        try:
            page.wait_for_url("https://github.com/", timeout=0) # Wait indefinitely until redirected to home
            print("✅ Login detected!")
        except Exception as e:
            print("Login check failed or timed out.")
            
        # Navigate to repository projects
        print(f"Navigating to {repo_url}/projects?type=beta")
        page.goto(f"{repo_url}/projects?type=beta")
        
        # Click "New project"
        try:
            # Select "New project" or "Link a project" -> "New project"
            # This selector path might vary, trying robust locators
            page.get_by_role("button", name="New project").first.click()
            print("Clicked 'New project'")
        except:
            print("Could not find 'New project' button directly. You might need to click it manually.")

        # Select "Board" template
        try:
            page.get_by_text("Board", exact=True).click()
            print("Selected 'Board' template")
            page.get_by_role("button", name="Create").click()
            print("Clicked Create")
        except:
            print("Could not select template automatically.")

        # Wait for project to load
        page.wait_for_load_state("networkidle")
        
        # Rename Project
        try:
            # Click title to rename
            page.get_by_test_id("project-title-input").click()
            page.get_by_test_id("project-title-input").fill("PAROL6 ros2_control Migration")
            page.keyboard.press("Enter")
            print("✅ Renamed project to 'PAROL6 ros2_control Migration'")
        except:
            print("Could not rename project. Please rename manually.")

        # Add automation to pause
        print("\n✨ Automation stopped here to allow manual column configuration.")
        print("Please configure columns:")
        print("1. Rename 'Todo' -> '📋 Backlog'")
        print("2. Rename 'In Progress' -> '🔄 In Progress'")
        print("3. Rename 'Done' -> '✅ Done'")
        print("4. Add '🐛 Blocked'")
        
        input("\nPress Enter to close browser when done...")
        browser.close()

if __name__ == "__main__":
    setup_project_board()
