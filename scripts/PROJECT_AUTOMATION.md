# PAROL6 Project Automation

This automation maintains a structured engineering project board using GitHub Projects V2.

---

## 🎯 Goals

- Enforce consistent task taxonomy
- Eliminate manual board maintenance
- Enable reproducible project structure for thesis evidence
- Automatically classify new tasks
- Scale safely for future development

---

## 🧱 Managed Fields

### Domain
Engineering responsibility.

- hardware
- esp32
- control
- foc
- ros
- vision
- robotics
- ai
- other

---

### Layer
System architecture layer.

- hardware
- firmware
- middleware
- control
- perception
- planning
- integration
- tooling
- documentation

---

### Maturity
Engineering readiness.

- idea
- prototype
- validated
- integrated
- production
- deprecated

Default = `idea`

---

## 🤖 Automatic Classification

Issues are classified based on keywords in the title.

### Example

| Title Contains | Domain | Layer |
|----------------|--------|-------|
| esp32, uart | esp32 | firmware |
| foc | foc | control |
| ros | ros | middleware |
| vision, yolo | vision | perception |
| robot | robotics | planning |
| model, dataset | ai | perception |
| motor, driver | hardware | hardware |
| otherwise | other | tooling |

---

## 🚀 How To Run

```bash
gh auth login
chmod +x scripts/setup_master_board.py
./scripts/setup_master_board.py
```

Safe to run multiple times.

### ➕ Adding New Tasks

Simply create a GitHub issue.

Automation will:
1. Detect it
2. Assign Domain
3. Assign Layer
4. Assign Maturity

No manual board edits required.

### ➕ Extending Rules

Edit inside `scripts/setup_master_board.py`:
- `DOMAIN_KEYWORDS`
- `DOMAIN_TO_LAYER`
- `FIELDS`

Then re-run the script.

### 🔒 Security

- No tokens stored in code
- Uses local gh authentication
- No browser automation
- No credentials logged

### 📚 Thesis Usage

This automation provides:
- Deterministic project management
- Traceable engineering workflow
- Evidence of professional tooling
- Reproducible environment

---

## 👁️ View Generation

We provide a helper script to generate standard engineering view filters.

### How to Create Saved Views

1. Run the generator:
   ```bash
   chmod +x scripts/generate_views.py
   ./scripts/generate_views.py
   ```
2. **Click** the link for the view you want (e.g., "Hardware").
3. In GitHub Projects, click the arrow next to the current view name.
4. Select **"Save view"** (or "Save changes to new view").
5. Name it (e.g., "Hardware").

This process must be done manually once per view, as the API does not support creating saved views programmatically.

