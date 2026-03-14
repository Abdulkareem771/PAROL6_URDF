# 🤖 SYSTEM OVERVIEW: WELDVISION MODELS TESTING TOOL
**CRITICAL CONTEXT FOR FUTURE AI ASSISTANTS**

If you are an AI reading this file, you have been instantiated to assist the User with their PAROL6 Robotics/Vision codebase. **DO NOT deviate from the established architectural patterns and user preferences documented below.**

## 1. Operating Environment & File System Mounts
- **Dockerized Execution:** The standard operating procedure is to run everything inside a Docker container named `parol6_dev` or `parol6-ultimate`. 
- **Filesystem Mapping:** The container utilizes volume mounts. The User's local Host OS directory (`/home/kareem`) is mounted inside the container at `/host_home`. **ALWAYS** ensure file dialogs or output directories refer to universally accessible paths (like `~` expanding properly) so outputs aren't trapped inside the container.
- **X11 Forwarding:** The GUI uses X11 forwarding (`DISPLAY=:1` or similar). Avoid operations that assume a native window manager is available. 

## 2. Strict UI/GUI Preferences (Tkinter)
The User has highly specific formatting preferences for Tkinter interfaces. **You must adhere to these** to avoid degrading the tool's user experience:
1. **"V1" Layout Pattern:** All custom GUIs must utilize the exact layout found in `yolo_training/yolo_gui.py` and `resunet_training/weld_seam_gui.py`.
   - **Dark Theme Sidebar:** A fixed-width dark sidebar on the left (`#1e2229` background).
   - **Single View Canvas:** A massive, singular canvas on the right. 
   - **NO Multicolumn/Grid Chaos:** Do *not* stack properties into wide horizontal strips above the image. Keep all controls restricted to vertical stacks inside the left sidebar.
2. **Hex Colors:** Do *NOT* use 8-character hex codes (e.g., `#RRGGBBAA`). Tkinter under this X11 configuration will crash. Only use standard 6-character hex codes (`#RRGGBB`).
3. **Sliders over Reruns:** Any parameter that affects the visual output (e.g., YOLO confidence threshold) should ideally be tied to a standard `tk.Scale` that instantly triggers a canvas redraw `_on_view_change()` without explicitly re-evaluating the heavy ML model.

## 3. The `xclip` Clipboard Workaround
Because the user operates within an X11-forwarded Docker container on a Linux host, standard Python OS clipboard libraries (`PIL.ImageGrab.grabclipboard()` or `pyperclip`) **will silently fail or return None** when asked for image data.

**The established, working solution** is invoking the `xclip` command-line utility via `subprocess`:
```python
proc = subprocess.run(["xclip", "-selection", "clipboard", "-t", "image/png", "-o"], capture_output=True)
with open("/tmp/parsed_image.png", "wb") as f:
    f.write(proc.stdout)
Image.open("/tmp/parsed_image.png").verify()
# Image is now safely grabbed from Host OS clipboard.
```
**Never** try to revert this back to `ImageGrab`. 

## 4. Subprocess Launching
The vision tools are united under `vision_work/launcher.py`. This script is meant to be extremely lightweight. If you add a new GUI tool to the repository, add a simple Tkinter Button to `launcher.py` that invokes it via `subprocess.run(["python3", "path/to/new_gui.py"])`.

## 5. Architectural Locations
- **Launcher:** `/vision_work/launcher.py`
- **ResUNet Seg Tester:** `/vision_work/resunet_training/weld_seam_gui.py`
- **YOLO Tester:** `/vision_work/yolo_training/yolo_gui.py`
- **ResUNet Model (Default pt):** `/vision_work/resunet_training/weld_seam_resunet_best.pt`
- **YOLO Model (Default pt):** `/vision_work/YOLO_resources/best.pt`
- **Human Documentation:** `/vision_work/docs/MODELS_TESTING_TOOL.md`
