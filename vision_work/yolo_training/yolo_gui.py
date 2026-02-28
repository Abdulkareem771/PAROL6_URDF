"""
WeldVision GUI — YOLO Object Detection Tester
=============================================
- Classic single-pane dark sidebar layout
- Standard OS file dialog for picking an image
- Paste from clipboard (Ctrl+V)
- Filter by target class tag
- Dropdown/Radio for View Mode (Original | Bounding Boxes | Mask Overlay | Cropped View)
- Live confidence threshold slider updates view without re-running
- Automated Batch Processing saving sequentially cropped results

Run: python3 yolo_gui.py
"""
import os, sys, threading, time, shutil, subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    sys.exit(1)

# ─── Colors ───────────────────────────────────────────────────────────────────
C = {
    "bg":     "#111318",
    "panel":  "#1e2229",
    "border": "#2d333b",
    "accent": "#58a6ff",   # Blue
    "green":  "#3fb950",
    "warn":   "#d29922",
    "err":    "#f85149",
    "text":   "#c9d1d9",
    "text2":  "#8b949e",
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WeldVision · YOLO Detector")
        self.geometry("1100x800")
        self.minsize(900, 600)
        self.configure(bg=C["bg"])

        # State
        self.model  = None
        self.image_path = tk.StringVar(value="")
        self.model_path = tk.StringVar(value=self._default_model())
        self.threshold  = tk.DoubleVar(value=0.5)
        self.view_mode  = tk.StringVar(value="Bounding Boxes")
        self.target_tag = tk.StringVar(value="")

        # Advanced State
        self.tag1 = tk.StringVar(value="")
        self.tag1_color = tk.StringVar(value="0,255,0") # Green
        self.tag2 = tk.StringVar(value="")
        self.tag2_color = tk.StringVar(value="255,100,0") # Orange
        
        self.batch_task = tk.StringVar(value="Crop Objects")

        # Data
        self._rgb  = None
        self._results = None
        self._current_view_rgb = None
        self._pil_img = None
        self._tk_img  = None

        self._build()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _default_model(self):
        # Look for a default best.pt in the current dir
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "best.pt")
        # if not found, check parent dir YOLO_resources
        if not os.path.exists(path):
            parent = os.path.dirname(os.path.dirname(here))
            alt_path = os.path.join(parent, "YOLO_resources", "best.pt")
            if os.path.exists(alt_path):
                return alt_path
        return path

    # ─── UI Building ──────────────────────────────────────────────────────────
    def _build(self):
        # Top Header
        hdr = tk.Frame(self, bg=C["panel"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text=" 🔍 WeldVision YOLO", font=("Segoe UI", 14, "bold"),
                 bg=C["panel"], fg=C["text"]).pack(side="left", padx=10)
        self._model_status = tk.Label(hdr, text="Loading model...", font=("Segoe UI", 10),
                                      bg=C["panel"], fg=C["warn"])
        self._model_status.pack(side="right", padx=15)

        # Main Split
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=4, pady=4)

        # Sidebar (Left) scrollable container
        sidebar_container = tk.Frame(main, bg=C["panel"], width=300)
        sidebar_container.pack(side="left", fill="y", padx=(0, 4))
        sidebar_container.pack_propagate(False)

        self.sidebar_canvas = tk.Canvas(sidebar_container, bg=C["panel"], highlightthickness=0)
        scrollbar = tk.Scrollbar(sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        
        sidebar = tk.Frame(self.sidebar_canvas, bg=C["panel"])
        
        sidebar.bind(
            "<Configure>",
            lambda e: self.sidebar_canvas.configure(
                scrollregion=self.sidebar_canvas.bbox("all")
            )
        )

        self.sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw", width=285)
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel scrolling bindings (Windows, Mac, and Linux)
        def _on_mousewheel(event):
            try:
                if event.delta:
                    self.sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except: pass
        self.sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.sidebar_canvas.bind_all("<Button-4>", lambda e: self.sidebar_canvas.yview_scroll(-1, "units"))
        self.sidebar_canvas.bind_all("<Button-5>", lambda e: self.sidebar_canvas.yview_scroll(1, "units"))

        # Notebook Setup
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=C["panel"], borderwidth=0)
        style.configure('TNotebook.Tab', background=C["border"], foreground=C["text"], padding=[10, 5], font=("Segoe UI", 9, "bold"))
        style.map('TNotebook.Tab', background=[('selected', C["accent"])], foreground=[('selected', '#ffffff')])

        self.notebook = ttk.Notebook(sidebar)
        self.notebook.pack(fill="both", expand=True, pady=(5, 0))
        
        tab_main = tk.Frame(self.notebook, bg=C["panel"])
        tab_adv = tk.Frame(self.notebook, bg=C["panel"])
        
        self.notebook.add(tab_main, text="Main Tester")
        self.notebook.add(tab_adv, text="Advanced Ops")

        # =========================================================
        # TAB 1: MAIN TESTER
        # =========================================================
        # -> Test Image
        self._section(tab_main, "Test Image")
        f1 = tk.Frame(tab_main, bg=C["panel"])
        f1.pack(fill="x", padx=12, pady=4)
        tk.Entry(f1, textvariable=self.image_path, bg=C["border"], fg=C["text"],
                 relief="flat", font=("Helvetica", 9), insertbackground=C["text"]).pack(fill="x", ipady=3)
        self._btn(f1, "📂 Browse Image...", self._browse_image).pack(fill="x", pady=(4, 0))
        self._btn(f1, "📋 Paste (Ctrl+V)", self._paste_image).pack(fill="x", pady=(4, 0))

        # -> Inference Action
        self._section(tab_main, "Inference")
        self._run_btn = tk.Button(tab_main, text="▶ Run Detection", command=self._run_async,
                                  bg=C["accent"], fg="#ffffff", font=("Segoe UI", 11, "bold"),
                                  relief="flat", pady=6, cursor="hand2")
        self._run_btn.pack(fill="x", padx=12, pady=8)

        # -> Controls
        self._section(tab_main, "Filters & View")
        f2 = tk.Frame(tab_main, bg=C["panel"])
        f2.pack(fill="x", padx=12, pady=2)
        
        tk.Label(f2, text="Target Tag (leave blank for all):", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 9)).pack(anchor="w")
        tk.Entry(f2, textvariable=self.target_tag, bg=C["border"], fg=C["text"],
                 relief="flat", font=("Helvetica", 10), insertbackground=C["text"]).pack(fill="x", ipady=3, pady=(0, 6))
        self.target_tag.trace_add("write", lambda *args: self._on_view_change())

        tk.Label(f2, text="Confidence Threshold:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 9)).pack(anchor="w")
        self._thr_lbl = tk.Label(f2, text="str(0.50)", bg=C["panel"], fg=C["accent"], font=("Segoe UI", 10, "bold"))
        self._thr_lbl.pack(anchor="e", pady=(0, 2))
        self._thr_lbl.config(text="0.50")
        
        tk.Scale(f2, from_=0.01, to=0.99, resolution=0.01, variable=self.threshold,
                 orient="horizontal", bg=C["panel"], fg=C["text"], troughcolor=C["border"],
                 highlightthickness=0, showvalue=False, command=self._on_thr_change).pack(fill="x")

        tk.Label(f2, text="Drawing Color (B,G,R):", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 9)).pack(anchor="w", pady=(6,0))
        self.draw_color = tk.StringVar(value="255,50,50") # Default BGR red-ish
        tk.Entry(f2, textvariable=self.draw_color, bg=C["border"], fg=C["text"],
                 relief="flat", font=("Helvetica", 10)).pack(fill="x", ipady=3, pady=(0, 6))
        self.draw_color.trace_add("write", lambda *args: self._on_view_change())

        # View Radios
        tk.Label(f2, text="View Mode:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 2))
        for mode in ["Original", "Bounding Boxes", "Mask Overlay", "Solid Color Mask", "Cropped View", "Dual Tag Mask"]:
            rb = tk.Radiobutton(f2, text=mode, variable=self.view_mode, value=mode,
                                bg=C["panel"], fg=C["text"], selectcolor=C["border"],
                                activebackground=C["panel"], activeforeground=C["text"],
                                font=("Segoe UI", 10), command=self._on_view_change)
            rb.pack(anchor="w", padx=6)

        # -> Model Path
        self._section(tab_main, "Model Weights")
        f3 = tk.Frame(tab_main, bg=C["panel"])
        f3.pack(fill="x", padx=12, pady=4)
        tk.Entry(f3, textvariable=self.model_path, bg=C["border"], fg=C["text"],
                 relief="flat", font=("Helvetica", 9), insertbackground=C["text"]).pack(fill="x", ipady=3)
        self._btn(f3, "Browse Model...", self._browse_model).pack(fill="x", pady=(4,0))

        # =========================================================
        # TAB 2: ADVANCED OPS
        # =========================================================
        
        # Current view saves
        self._section(tab_adv, "Quick Export")
        f_exp = tk.Frame(tab_adv, bg=C["panel"])
        f_exp.pack(fill="x", padx=12, pady=4)
        self._btn(f_exp, "💾 Save Raw Image", lambda: self._save_specific("raw")).pack(fill="x", pady=(0,4))
        self._btn(f_exp, "💾 Save Mask Only (B&W)", lambda: self._save_specific("mask")).pack(fill="x")
        
        # Dual Tag Setup
        self._section(tab_adv, "Dual Tag Setup")
        f_dual = tk.Frame(tab_adv, bg=C["panel"])
        f_dual.pack(fill="x", padx=12, pady=4)
        tk.Label(f_dual, text="Select 'Dual Tag Mask' View Mode in Main Tab to activate.", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8), wraplength=250, justify="left").pack(anchor="w", pady=(0, 6))
        
        tk.Label(f_dual, text="Tag 1 (Class Name):", bg=C["panel"], fg=C["text"], font=("Segoe UI", 9)).pack(anchor="w")
        tk.Entry(f_dual, textvariable=self.tag1, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 9)).pack(fill="x", ipady=3)
        tk.Label(f_dual, text="Tag 1 Color (B,G,R):", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(anchor="w")
        tk.Entry(f_dual, textvariable=self.tag1_color, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 9)).pack(fill="x", ipady=3, pady=(0,6))
        
        tk.Label(f_dual, text="Tag 2 (Class Name):", bg=C["panel"], fg=C["text"], font=("Segoe UI", 9)).pack(anchor="w")
        tk.Entry(f_dual, textvariable=self.tag2, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 9)).pack(fill="x", ipady=3)
        tk.Label(f_dual, text="Tag 2 Color (B,G,R):", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(anchor="w")
        tk.Entry(f_dual, textvariable=self.tag2_color, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 9)).pack(fill="x", ipady=3)
        
        self.tag1.trace_add("write", lambda *args: self._on_view_change())
        self.tag2.trace_add("write", lambda *args: self._on_view_change())
        self.tag1_color.trace_add("write", lambda *args: self._on_view_change())
        self.tag2_color.trace_add("write", lambda *args: self._on_view_change())

        # Batch Data Generation
        self._section(tab_adv, "Advanced Batch Gen")
        fb_adv = tk.Frame(tab_adv, bg=C["panel"])
        fb_adv.pack(fill="x", padx=12, pady=4)
        
        self.adv_batch_in  = tk.StringVar()
        self.adv_batch_out = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "weld_batch_export"))
        tk.Label(fb_adv, text="Input Folder:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(anchor="w")
        tk.Entry(fb_adv, textvariable=self.adv_batch_in, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 8)).pack(fill="x")
        self._btn(fb_adv, "Select Input...", lambda: self.adv_batch_in.set(filedialog.askdirectory(title="Input") or self.adv_batch_in.get())).pack(fill="x", pady=(2,4))
        
        tk.Label(fb_adv, text="Output Folder:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(anchor="w")
        tk.Entry(fb_adv, textvariable=self.adv_batch_out, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 8)).pack(fill="x")
        self._btn(fb_adv, "Select Output...", lambda: self.adv_batch_out.set(filedialog.askdirectory(title="Output") or self.adv_batch_out.get())).pack(fill="x", pady=(2,8))

        tk.Label(fb_adv, text="Generation Task:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(anchor="w")
        self.batch_task_combo = ttk.Combobox(fb_adv, textvariable=self.batch_task, state="readonly", 
                                             values=["Crop Objects (Tag 1 & 2)", "Export YOLO Annotations (.txt)", "Export Dual Color Masks", "Export Binary Masks"])
        self.batch_task_combo.pack(fill="x", pady=(0, 8))

        self._adv_batch_btn = tk.Button(fb_adv, text="🚀 Auto-Run Task", command=self._run_advanced_batch_async,
                                  bg=C["warn"], fg="#ffffff", font=("Segoe UI", 10, "bold"),
                                  relief="flat", pady=4, cursor="hand2")
        self._adv_batch_btn.pack(fill="x")

        # -> Space spacer
        tk.Frame(sidebar, bg=C["panel"]).pack(fill="both", expand=True)

        # -> Save Result (Global)
        self._btn(sidebar, "💾 Save View As...", self._save_result, bg=C["green"], fg="#ffffff").pack(fill="x", padx=12, pady=12)

        # View Area (Right)
        view_frame = tk.Frame(main, bg=C["border"])
        view_frame.pack(side="left", fill="both", expand=True)

        # Info bar above canvas
        self._info_bar = tk.Frame(view_frame, bg=C["border"], height=30)
        self._info_bar.pack(fill="x")
        self._info_bar.pack_propagate(False)
        self._lbl_title = tk.Label(self._info_bar, text="No Image", bg=C["border"], fg=C["text"], font=("Segoe UI", 10, "bold"))
        self._lbl_title.pack(side="left", padx=10)
        self._lbl_stats = tk.Label(self._info_bar, text="", bg=C["border"], fg=C["text2"], font=("Segoe UI", 9))
        self._lbl_stats.pack(side="right", padx=10)

        # Canvas
        self.canvas = tk.Canvas(view_frame, bg=C["bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        
        self.bind("<Return>", lambda e: self._run_async())
        self.bind("<Control-v>", lambda e: self._paste_image())


    def _section(self, parent, text):
        f = tk.Frame(parent, bg=C["panel"])
        f.pack(fill="x", padx=8, pady=(15, 2))
        tk.Label(f, text=text.upper(), font=("Segoe UI", 9, "bold"),
                 fg=C["text2"], bg=C["panel"]).pack(side="left")

    def _btn(self, parent, text, cmd, bg=C["border"], fg=C["text"]):
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                      relief="flat", font=("Segoe UI", 10), cursor="hand2", pady=4, borderwidth=0)
        return b

    # ─── Callbacks ────────────────────────────────────────────────────────────
    def _browse_image(self):
        p = filedialog.askopenfilename(title="Select test image",
                                       filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.webp"), ("All", "*.*")])
        if p:
            self.image_path.set(p)
            self._lbl_title.config(text=f"{os.path.basename(p)}")
            if self.model: self._run_async()

    def _paste_image(self):
        try:
            os.makedirs("/tmp/weldvision", exist_ok=True)
            tmp_path = "/tmp/weldvision/pasted_image.png"
            
            proc = subprocess.run(
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
                capture_output=True
            )
            
            if proc.returncode != 0 or not proc.stdout:
                messagebox.showinfo("Clipboard", "No image found in clipboard.")
                return
                
            with open(tmp_path, "wb") as f:
                f.write(proc.stdout)
            
            Image.open(tmp_path).verify()
            
            self.image_path.set(tmp_path)
            self._lbl_title.config(text="Pasted Image")
            if self.model: self._run_async()
            
        except FileNotFoundError:
            messagebox.showerror("Dependency Missing", "xclip is not installed.")
        except Exception as e:
            messagebox.showerror("Paste Error", f"Clipboard content is not a valid image.\n\n{e}")

    def _browse_model(self):
        p = filedialog.askopenfilename(title="Select YOLO model", filetypes=[("PT", "*.pt"), ("All", "*.*")])
        if p:
            self.model_path.set(p)
            threading.Thread(target=self._load_model, daemon=True).start()

    def _on_thr_change(self, val):
        self._thr_lbl.config(text=f"{float(val):.2f}")
        # View change filters by threshold dynamically
        self._on_view_change()

    def _filter_results(self):
        if not self._results: return []
        boxes = self._results[0].boxes
        if not boxes: return []
        
        thr = self.threshold.get()
        target = self.target_tag.get().strip().lower()
        names = self._results[0].names
        
        filtered = []
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < thr:
                continue
            cls_name = names[int(boxes.cls[i])].lower()
            if target and target not in cls_name:
                continue
            filtered.append(i)
        return filtered

    def _on_view_change(self):
        if self._rgb is None or not self._results: return
        v = self.view_mode.get()
        
        indices = self._filter_results()
        names = self._results[0].names
        boxes = self._results[0].boxes
        masks = self._results[0].masks
        
        # Parse selected BGR color
        try:
            c_str = self.draw_color.get().strip()
            b, g, r = [int(x.strip()) for x in c_str.split(',')]
            color_bgr = (b, g, r)               # For OpenCV shapes
            color_rgb = np.array([r, g, b])     # For numpy mask operations (since self._rgb is RGB)
        except:
            color_bgr = (255, 50, 50)
            color_rgb = np.array([255, 50, 50])
        
        img_copy = self._rgb.copy()
        
        if v == "Original":
            self._current_view_rgb = img_copy
        elif v == "Bounding Boxes":
            for i in indices:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                label = f"{names[cls]} {conf:.2f}"
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_rgb.tolist(), 2)
                cv2.putText(img_copy, label, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._current_view_rgb = img_copy
        elif v == "Mask Overlay":
            if masks is not None:
                # YOLO segmentation mask
                for i in indices:
                    mask = masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (img_copy.shape[1], img_copy.shape[0]))
                    img_copy[mask > 0.5] = img_copy[mask > 0.5] * 0.5 + color_rgb * 0.5
            else:
                # fallback if not a segmentation model
                self._current_view_rgb = img_copy # Just fallback
                
            for i in indices:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_rgb.tolist(), 2)
                
            self._current_view_rgb = img_copy
        elif v == "Solid Color Mask":
            if masks is not None:
                for i in indices:
                    mask = masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (img_copy.shape[1], img_copy.shape[0]))
                    img_copy[mask > 0.5] = color_rgb
            else:
                for i in indices:
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_rgb.tolist(), -1)
            self._current_view_rgb = img_copy
        elif v == "Cropped View":
            if len(indices) > 0:
                # Crop to the first match
                i = indices[0]
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                # Provide a bit of padding
                p = 10
                h, w = img_copy.shape[:2]
                x1, y1 = max(0, x1-p), max(0, y1-p)
                x2, y2 = min(w, x2+p), min(h, y2+p)
                self._current_view_rgb = self._rgb[y1:y2, x1:x2]
            else:
                self._current_view_rgb = img_copy
        
        elif v == "Dual Tag Mask":
            t1 = self.tag1.get().strip().lower()
            t2 = self.tag2.get().strip().lower()
            
            def parse_c(color_str, def_bgr):
                try:
                    b, g, r = [int(x.strip()) for x in color_str.split(',')]
                    return (b, g, r), np.array([r, g, b])
                except:
                    return def_bgr, np.array([def_bgr[2], def_bgr[1], def_bgr[0]])
                    
            c1_b, c1_r = parse_c(self.tag1_color.get(), (0, 255, 0))
            c2_b, c2_r = parse_c(self.tag2_color.get(), (0, 100, 255))
            
            matched_count = 0
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf < self.threshold.get(): continue
                cls_name = names[int(boxes.cls[i])].lower()
                
                if t1 and t1 in cls_name:
                    c_rgb, c_bgr = c1_r, c1_b
                    matched_count += 1
                elif t2 and t2 in cls_name:
                    c_rgb, c_bgr = c2_r, c2_b
                    matched_count += 1
                else:
                    continue
                    
                if masks is not None:
                    mask = masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (img_copy.shape[1], img_copy.shape[0]))
                    img_copy[mask > 0.5] = c_rgb
                else:
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), c_bgr, -1)
            self._current_view_rgb = img_copy
            indices = [0] * matched_count # Hack to update the stats text label accurately
            
        # Stats update
        self._lbl_stats.config(text=f"Detected: {len(indices)} matches")

        # Redraw
        self._pil_img = Image.fromarray(self._current_view_rgb)
        self._redraw()

    def _save_result(self):
        if self._current_view_rgb is None:
            messagebox.showinfo("Wait", "Run inference first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile=f"yolo_{os.path.basename(self.image_path.get())}.png",
                                         filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if p:
            bgr = cv2.cvtColor(self._current_view_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, bgr)
            messagebox.showinfo("Saved", f"Currently viewed image saved to:\n{p}")

    def _save_specific(self, mode):
        if self._rgb is None:
            messagebox.showinfo("Wait", "Run inference first.")
            return
            
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"yolo_{mode}_{os.path.basename(self.image_path.get())}.png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
        )
        if not p: return
        
        if mode == "raw":
            bgr = cv2.cvtColor(self._rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, bgr)
        elif mode == "mask":
            mask_img = np.zeros(self._rgb.shape[:2], dtype=np.uint8)
            if self._results and self._results[0].masks is not None:
                masks = self._results[0].masks
                indices = self._filter_results()
                for i in indices:
                    m = masks.data[i].cpu().numpy()
                    m = cv2.resize(m, (mask_img.shape[1], mask_img.shape[0]))
                    mask_img[m > 0.5] = 255
            else:
                messagebox.showwarning("Warning", "No segmentation masks found. A blank image was saved.")
            cv2.imwrite(p, mask_img)
            
        messagebox.showinfo("Saved", f"{mode.capitalize()} image saved to:\n{p}")

    # ─── Async Logic ──────────────────────────────────────────────────────────
    def _load_model(self):
        p = self.model_path.get()
        if not os.path.exists(p):
            self.after(0, lambda: self._model_status.config(text="Model File Missing", fg=C["err"]))
            return
        self.after(0, lambda: self._model_status.config(text="Loading YOLO...", fg=C["warn"]))
        try:
            self.model = YOLO(p)
            self.after(0, lambda: self._model_status.config(text="YOLO Ready", fg=C["green"]))
        except Exception as e:
            self.after(0, lambda: self._model_status.config(text=f"Error: {e}", fg=C["err"]))

    def _run_async(self):
        img_path = self.image_path.get()
        if not self.model or not img_path or not os.path.exists(img_path):
            messagebox.showwarning("Incomplete", "Make sure a test image is selected and the model is loaded.")
            return
        
        self._run_btn.config(state="disabled", text="Running...")
        self._lbl_title.config(text=f"Processing {os.path.basename(img_path)}...")
        
        def task():
            t0 = time.time()
            try:
                bgr = cv2.imread(img_path)
                if bgr is None: raise ValueError("Invalid image")
                
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
                # Run YOLO (low conf here, we filter manually later so the slider works live)
                # Setting conf=0.01 to get all proposals, enabling dynamic filtering by the slider
                results = self.model(bgr, conf=0.01, verbose=False)
                
                self._rgb = rgb
                self._results = results
                
                ms = (time.time() - t0) * 1000
                self.after(0, self._lbl_title.config, {"text": f"{os.path.basename(img_path)} ({ms:.1f}ms)"})
                self.after(0, self._on_view_change)
            except Exception as e:
                self.after(0, messagebox.showerror, "Error", str(e))
                self.after(0, self._lbl_title.config, {"text": "Error running inference"})
            finally:
                self.after(0, self._run_btn.config, {"state": "normal", "text": "▶ Run Detection"})
        
        threading.Thread(target=task, daemon=True).start()

    def _run_batch_async(self):
        in_d = self.batch_in.get()
        out_d = self.batch_out.get()
        if not in_d or not out_d or not os.path.exists(in_d):
            messagebox.showwarning("Batch", "Select valid input and output folders first.")
            return
            
        if not self.model:
            messagebox.showwarning("Batch", "Load a model first.")
            return

        self._batch_btn.config(state="disabled", text="Running Batch...")
        
        def task():
            images = [f for f in os.listdir(in_d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not images:
                self.after(0, messagebox.showinfo, "Batch", "No images found in input folder.")
                self.after(0, self._batch_btn.config, {"state": "normal", "text": "🚀 Auto-Run Batch"})
                return

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            batch_out_dir = os.path.join(out_d, f"yolo_results_{timestamp}")
            os.makedirs(batch_out_dir, exist_ok=True)
            
            target = self.target_tag.get().strip().lower()
            thr = self.threshold.get()
            
            count = 0
            for img_name in sorted(images):
                try:
                    p = os.path.join(in_d, img_name)
                    bgr = cv2.imread(p)
                    if bgr is None: continue
                    
                    results = self.model(bgr, conf=thr, verbose=False)
                    res = results[0]
                    boxes = res.boxes
                    if not boxes: continue
                    
                    # Filter and crop
                    for i in range(len(boxes)):
                        cls_name = res.names[int(boxes.cls[i])].lower()
                        if target and target not in cls_name:
                            continue
                            
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                        pd = 10 # padding
                        h, w = bgr.shape[:2]
                        x1, y1 = max(0, x1-pd), max(0, y1-pd)
                        x2, y2 = min(w, x2+pd), min(h, y2+pd)
                        
                        crop = bgr[y1:y2, x1:x2]
                        # Save it sequentially
                        count += 1
                        out_name = f"{count:04d}_{img_name}"
                        cv2.imwrite(os.path.join(batch_out_dir, out_name), crop)
                        
                except Exception as e:
                    print(f"Error on {img_name}: {e}")
            
            self.after(0, messagebox.showinfo, "Batch Complete", f"Processed {len(images)} images.\nSaved {count} cropped detections to:\n{batch_out_dir}")
            self.after(0, self._batch_btn.config, {"state": "normal", "text": "🚀 Auto-Run Batch"})
            
        threading.Thread(target=task, daemon=True).start()

    def _run_advanced_batch_async(self):
        in_d = self.adv_batch_in.get()
        out_d = self.adv_batch_out.get()
        if not in_d or not out_d or not os.path.exists(in_d):
            messagebox.showwarning("Batch", "Select valid input and output folders first.")
            return
            
        if not self.model:
            messagebox.showwarning("Batch", "Load a model first.")
            return

        self._adv_batch_btn.config(state="disabled", text="Running Task...")
        
        task_type = self.batch_task.get()
        t1 = self.tag1.get().strip().lower()
        t2 = self.tag2.get().strip().lower()
        thr = self.threshold.get()
        
        def parse_c(color_str, def_bgr):
            try:
                b, g, r = [int(x.strip()) for x in color_str.split(',')]
                return (b, g, r), np.array([r, g, b])
            except:
                return def_bgr, np.array([def_bgr[2], def_bgr[1], def_bgr[0]])
                
        c1_b, c1_r = parse_c(self.tag1_color.get(), (0, 255, 0))
        c2_b, c2_r = parse_c(self.tag2_color.get(), (0, 100, 255))
        
        def task():
            images = [f for f in os.listdir(in_d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not images:
                self.after(0, messagebox.showinfo, "Batch", "No images found in input folder.")
                self.after(0, self._adv_batch_btn.config, {"state": "normal", "text": "🚀 Auto-Run Task"})
                return

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            batch_out_dir = os.path.join(out_d, f"yolo_adv_{timestamp}")
            os.makedirs(batch_out_dir, exist_ok=True)
            
            count = 0
            for img_name in sorted(images):
                try:
                    p = os.path.join(in_d, img_name)
                    bgr = cv2.imread(p)
                    if bgr is None: continue
                    img_h, img_w = bgr.shape[:2]
                    
                    results = self.model(bgr, conf=thr, verbose=False)
                    res = results[0]
                    boxes = res.boxes
                    if not boxes: continue
                    
                    # Core task implementation
                    if task_type == "Crop Objects (Tag 1 & 2)":
                        for i in range(len(boxes)):
                            cls_name = res.names[int(boxes.cls[i])].lower()
                            if t1 and t1 in cls_name: pass
                            elif t2 and t2 in cls_name: pass
                            elif not t1 and not t2: pass # crop all if both blank
                            else: continue
                                
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                            pd = 10
                            x1, y1 = max(0, x1-pd), max(0, y1-pd)
                            x2, y2 = min(img_w, x2+pd), min(img_h, y2+pd)
                            crop = bgr[y1:y2, x1:x2]
                            
                            count += 1
                            out_name = f"{count:04d}_{img_name}"
                            cv2.imwrite(os.path.join(batch_out_dir, out_name), crop)
                            
                    elif task_type == "Export YOLO Annotations (.txt)":
                        base_name = os.path.splitext(img_name)[0]
                        txt_path = os.path.join(batch_out_dir, f"{base_name}.txt")
                        lines = []
                        for i in range(len(boxes)):
                            cls_name = res.names[int(boxes.cls[i])].lower()
                            cls_id = -1
                            if t1 and t1 in cls_name: cls_id = 0
                            elif t2 and t2 in cls_name: cls_id = 1
                            elif not t1 and not t2: cls_id = int(boxes.cls[i])
                            
                            if cls_id == -1: continue
                            
                            # YOLO format: class x_center y_center width height (normalized)
                            x1, y1, x2, y2 = map(float, boxes.xyxy[i])
                            xc = ((x1 + x2) / 2) / img_w
                            yc = ((y1 + y2) / 2) / img_h
                            w = (x2 - x1) / img_w
                            h = (y2 - y1) / img_h
                            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            
                        if lines:
                            with open(txt_path, "w") as f:
                                f.write("\n".join(lines))
                            count += 1
                            
                    elif task_type in ["Export Dual Color Masks", "Export Binary Masks"]:
                        mask_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                        drawn = False
                        for i in range(len(boxes)):
                            cls_name = res.names[int(boxes.cls[i])].lower()
                            if t1 and t1 in cls_name:
                                c_bgr = c1_b if task_type == "Export Dual Color Masks" else (255, 255, 255)
                            elif t2 and t2 in cls_name:
                                c_bgr = c2_b if task_type == "Export Dual Color Masks" else (255, 255, 255)
                            elif not t1 and not t2:
                                c_bgr = (255, 255, 255)
                            else:
                                continue
                                
                            drawn = True
                            if res.masks is not None:
                                m = res.masks.data[i].cpu().numpy()
                                m = cv2.resize(m, (img_w, img_h))
                                mask_img[m > 0.5] = c_bgr
                            else:
                                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                                cv2.rectangle(mask_img, (x1, y1), (x2, y2), c_bgr, -1)
                        
                        if drawn:
                            count += 1
                            out_name = f"{count:04d}_{img_name}"
                            cv2.imwrite(os.path.join(batch_out_dir, out_name), mask_img)

                except Exception as e:
                    print(f"Error on {img_name}: {e}")
            
            self.after(0, messagebox.showinfo, "Batch Complete", f"Processed {len(images)} images.\nGenerated {count} files in:\n{batch_out_dir}")
            self.after(0, self._adv_batch_btn.config, {"state": "normal", "text": "🚀 Auto-Run Task"})
            
        threading.Thread(target=task, daemon=True).start()

    # ─── Draw ─────────────────────────────────────────────────────────────────
    def _redraw(self):
        if self._pil_img is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        ratio = min(cw / self._pil_img.width, ch / self._pil_img.height)
        nw, nh = int(self._pil_img.width * ratio), int(self._pil_img.height * ratio)
        if nw > 0 and nh > 0:
            resized = self._pil_img.resize((nw, nh), Image.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(resized)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._tk_img)

    def _on_resize(self, event):
        self._redraw()


if __name__ == "__main__":
    App().mainloop()
