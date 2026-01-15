# PAROL6 Project Views

This document defines the standardized task views used for engineering management.

All views are backed by automated classification fields.

---

## Domain Views

| View | Filter |
|--------|-----------|
| Hardware | Domain:hardware |
| ESP32 | Domain:esp32 |
| Control | Domain:control |
| FOC | Domain:foc |
| Vision | Domain:vision |
| ROS | Domain:ros |
| Robotics | Domain:robotics |
| AI | Domain:ai |
| Other | Domain:other |

---

## Layer Views

| View | Filter |
|--------|-----------|
| Firmware | Layer:firmware |
| Perception | Layer:perception |
| Planning | Layer:planning |
| Integration | Layer:integration |

---

## Maturity Views

| View | Filter |
|--------|-----------|
| Prototype | Maturity:prototype |
| Validated | Maturity:validated |
| Integrated | Maturity:integrated |

---

## Automation

All tasks are automatically classified using:

`scripts/setup_master_board.py`

New issues automatically appear in all relevant views.

---

## Extension

To add a new domain or layer:

1. Edit FIELDS in `setup_master_board.py`
2. Add classification keywords
3. Re-run automation
4. Create new saved view

No migration needed.
