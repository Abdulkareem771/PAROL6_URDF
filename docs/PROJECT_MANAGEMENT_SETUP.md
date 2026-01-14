# Project Management Setup Guide

**Recommendation**: For a graduation project (Thesis), **Notion** is often the best balance of documentation and task management.

## 🛠️ Tool Comparison

| Feature | Notion ⭐️ | GitHub Projects | Trello |
| :--- | :--- | :--- | :--- |
| **Best For** | Docs + Wiki + Tasks | Pure Code Tasks | Simple Lists |
| **Pros** | All-in-one workspace | Linked to Code/PRs | Very simple |
| **Cons** | Manual sync with code | No "Wiki" features | Limited features |
| **Verdict** | **Winner** for Teams | Good for Dev-only | Too basic |

---

## 🚀 How to Setup Notion for PAROL6

1. **Create a "Teamspace"** called "PAROL6 Thesis".
2. **Create a Database** page called "Task Board".
3. **Add Columns (Properties)**:
   - `Status`: To Do, In Progress, Review, Done
   - `Assignee`: Person (Teammate)
   - `Priority`: High, Medium, Low
   - `Phase`: Select (e.g., Phase 1: Parallel, Phase 2: Integration)
   - `Due Date`: Date

---

## ⚡ Fast-Track: Import Tasks via CSV

Don't type everything manually! 
1. Copy the code block below.
2. Save it as `parol6_tasks.csv` on your computer.
3. In Notion: Click `...` > `Merge with CSV` (or Import) to instantly create cards.

```csv
Name,Status,Assignee,Priority,Phase,Description
"Teammate A: Collect workpiece dataset",In Progress,Teammate A,High,Phase 1,"Collect ~100 images of the workpiece from different angles"
"Teammate A: Annotate Dataset",To Do,Teammate A,High,Phase 1,"Label images using Roboflow or CVAT"
"Teammate A: Train Custom YOLO",To Do,Teammate A,High,Phase 1,"Train YOLOv8 on the annotated dataset"
"Teammate B: Create YOLO Detector Node",To Do,Teammate B,High,Phase 1,"Implement ROS 2 node using generic yolov8n.pt"
"Teammate B: Create Depth Matcher Node",To Do,Teammate B,High,Phase 1,"Map 2D bounding boxes to 3D pointcloud coordinates"
"Teammate B: Testing & Validation",To Do,Teammate B,Medium,Phase 1,"Verify depth accuracy with ruler/measurement"
"Integrate: Swap vs Custom Model",To Do,Kareem,High,Phase 2,"Replace generic model with Teammate A's trained model"
"Integrate: Seam Extractor Node",To Do,Kareem,High,Phase 2,"Implement logic to find weld line within bounding box"
"System: Path Generation (B-Spline)",To Do,Kareem,High,Phase 3,"Generate smooth trajectory from seam points"
"System: MoveIt Execution",To Do,Kareem,High,Phase 3,"Execute path on real robot"
"Documentation: Thesis Report",To Do,All,Medium,Phase 3,"Final documentation for graduation"
```

## 🧠 Workflow Advice

1. **Weekly Sprints**: Move 3-5 cards to "In Progress" each Monday.
2. **Docs Integration**: Since you use Notion, you can create a page for each major task (like "Depth Matcher") and link it in the card.
3. **GitHub Link**: You can paste GitHub PR links into the Notion card comments.

This keeps "Coding" in VS Code/GitHub and "Planning/Docs" in Notion.
