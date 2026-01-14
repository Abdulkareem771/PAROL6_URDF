# GitHub Projects Setup Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Create a New Project
1. Go to your GitHub repository
2. Click the **"Projects"** tab
3. Click **"New project"**
4. Choose **"Board"** template
5. Name it: **"PAROL6 ros2_control Migration"**

### Step 2: Customize Columns
Rename the default columns to:
- **📋 Backlog** (To Do)
- **🔄 In Progress**
- **✅ Done**
- **🐛 Blocked**

### Step 3: Create Issues from Templates
1. Go to **Issues** tab
2. Click **"New issue"**
3. You'll see templates for:
   - 🚀 Day 1 - SIL Validation
   - 📡 Day 2 - Serial TX
   - 🔄 Day 3 - Feedback Loop
   - 🤖 Day 4 - First Motion
   - 🎓 Day 5 - Validation & Thesis
   - 🐛 Bug Report

4. Create issues for Days 1-5
5. They'll automatically be added to your project board

### Step 4: Configure Automation
1. In your project, click **"..."** (top right)
2. Select **"Workflows"**
3. Enable these automations:
   - **Auto-add new issues** to project
   - **Move to "In Progress"** when issue is assigned
   - **Move to "Done"** when issue is closed
   - **Move PRs** to appropriate columns

### Step 5: Set Up Custom Fields (Optional but Recommended)
1. Click **"..."** → **"Settings"**
2. Add custom fields:
   - **Priority:** Single select (High, Medium, Low)
   - **Phase:** Single select (Day 1, Day 2, Day 3, Day 4, Day 5)
   - **Estimated Hours:** Number
   - **Validation Status:** Single select (Pending, Pass, Fail)

---

## 📊 Example Project Board Structure

```
┌─────────────────────────────────────────────────────────────┐
│  PAROL6 ros2_control Migration                              │
├───────────────┬──────────────┬──────────┬──────────────────┤
│  📋 Backlog   │ 🔄 In Prog   │ ✅ Done  │  🐛 Blocked      │
├───────────────┼──────────────┼──────────┼──────────────────┤
│ [DAY-3]       │ [DAY-2]      │ [DAY-1]  │                  │
│ Feedback      │ Serial TX    │ SIL ✅   │                  │
│               │              │          │                  │
│ [DAY-4]       │              │          │                  │
│ First Motion  │              │          │                  │
│               │              │          │                  │
│ [DAY-5]       │              │          │                  │
│ Validation    │              │          │                  │
└───────────────┴──────────────┴──────────┴──────────────────┘
```

---

## 🤖 Automation Features

The included GitHub Action (`.github/workflows/project-automation.yml`) provides:

### ✅ Auto-add Issues
- New issues are automatically added to the project board
- Issues are labeled by phase (Day 1-5)

### ✅ Status Updates
- Issues move to "In Progress" when assigned
- Issues move to "Done" when closed
- PRs are linked to related issues

### ✅ Day 1 Special Handling
- When Day 1 issue is created, it auto-closes with validation results
- Adds completion comment with metrics

---

## 📋 Using the Board Daily

### For You (Project Lead)
```bash
# Create Day 2 issue
gh issue create --template day2-serial-tx.md

# Assign to teammate
gh issue edit 2 --add-assignee teammate-username

# Link PR to issue
gh pr create --title "Implement serial TX" --body "Closes #2"
```

### For Teammates
1. **Check assigned issues:**
   - Filter: `assignee:@me is:open`
2. **Update progress:**
   - Move card to "In Progress"
   - Comment on blockers
3. **Complete task:**
   - Create PR linking issue
   - Issue auto-closes on merge

---

## 🎯 Integrations

### Link Commits to Issues
Use keywords in commit messages:
```bash
git commit -m "Implement write() method - fixes #2"
git commit -m "Add timing guards - relates to #2"
```

### Link PRs to Issues
In PR description:
```markdown
Closes #2
Implements serial TX with non-blocking I/O
```

---

## 📱 Mobile Access

GitHub Projects works on mobile:
- **iOS/Android:** Use GitHub mobile app
- **Web:** Fully responsive on mobile browsers

---

## 👥 Team Collaboration

### Invite Teammates
1. Go to repository **Settings** → **Collaborators**
2. Add teammates by username
3. They'll see the project automatically

### Assign Tasks
- Drag issue to "In Progress"
- Click issue → **Assignees** → Select teammate
- They get notified

---

## 📈 Tracking Progress

### View Insights
1. Click **"Insights"** in project
2. See:
   - Issue velocity
   - Completion rate
   - Phase distribution
   - Time in progress

### Generate Reports
- Export to CSV for thesis appendix
- Screenshot board for presentations

---

## 🎓 Thesis Benefits

Your examiners can see:
- ✅ **Project timeline** - When tasks were completed
- ✅ **Work allocation** - Who did what
- ✅ **Issue tracking** - Problems encountered and solved
- ✅ **Documentation** - Linked to issues for traceability

**Pro tip:** Keep your project board public to show in thesis defense!

---

## 🔧 Troubleshooting

### Issues not appearing in project?
- Check issue has correct label (phase-1, phase-2, etc.)
- Manually add via project board "+" button

### Automation not working?
- Go to **Actions** tab → Check workflow runs
- Verify GitHub token permissions

### Need help?
- GitHub Docs: https://docs.github.com/en/issues/planning-and-tracking-with-projects
- GitHub Community: https://github.community

---

**Status:** ✅ Ready to use  
**Next:** Create your project and issues!
