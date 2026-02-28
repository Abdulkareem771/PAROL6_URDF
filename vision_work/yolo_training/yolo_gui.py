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

        # Sidebar (Left)
        sidebar = tk.Frame(main, bg=C["panel"], width=280)
        sidebar.pack(side="left", fill="y", padx=(0, 4))
        sidebar.pack_propagate(False)

        # -> Test Image
        self._section(sidebar, "Test Image")
        f1 = tk.Frame(sidebar, bg=C["panel"])
        f1.pack(fill="x", padx=12, pady=4)
        tk.Entry(f1, textvariable=self.image_path, bg=C["border"], fg=C["text"],
                 relief="flat", font=("Helvetica", 9), insertbackground=C["text"]).pack(fill="x", ipady=3)
        self._btn(f1, "📂 Browse Image...", self._browse_image).pack(fill="x", pady=(4, 0))
        self._btn(f1, "📋 Paste (Ctrl+V)", self._paste_image).pack(fill="x", pady=(4, 0))

        # -> Inference Action
        self._section(sidebar, "Inference")
        self._run_btn = tk.Button(sidebar, text="▶ Run Detection", command=self._run_async,
                                  bg=C["accent"], fg="#ffffff", font=("Segoe UI", 11, "bold"),
                                  relief="flat", pady=6, cursor="hand2")
        self._run_btn.pack(fill="x", padx=12, pady=8)

        # -> Controls
        self._section(sidebar, "Filters & View")
        f2 = tk.Frame(sidebar, bg=C["panel"])
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

        # View Radios
        tk.Label(f2, text="View Mode:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 2))
        for mode in ["Original", "Bounding Boxes", "Mask Overlay", "Cropped View"]:
            rb = tk.Radiobutton(f2, text=mode, variable=self.view_mode, value=mode,
                                bg=C["panel"], fg=C["text"], selectcolor=C["border"],
                                activebackground=C["panel"], activeforeground=C["text"],
                                font=("Segoe UI", 10), command=self._on_view_change)
            rb.pack(anchor="w", padx=6)

        # -> Model Path
        self._section(sidebar, "Model Weights")
        f3 = tk.Frame(sidebar, bg=C["panel"])
        f3.pack(fill="x", padx=12, pady=4)
        tk.Entry(f3, textvariable=self.model_path, bg=C["border"], fg=C["text"],
                 relief="flat", font=("Helvetica", 9), insertbackground=C["text"]).pack(fill="x", ipady=3)
        self._btn(f3, "Browse Model...", self._browse_model).pack(fill="x", pady=(4,0))

        # -> Batch Processing
        self._section(sidebar, "Batch Cropping")
        fb = tk.Frame(sidebar, bg=C["panel"])
        fb.pack(fill="x", padx=12, pady=4)
        
        self.batch_in  = tk.StringVar()
        self.batch_out = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "weld_results"))
        tk.Label(fb, text="Input Folder:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(anchor="w")
        tk.Entry(fb, textvariable=self.batch_in, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 8)).pack(fill="x")
        self._btn(fb, "Select Input...", lambda: self.batch_in.set(filedialog.askdirectory(title="Input Folder") or self.batch_in.get())).pack(fill="x", pady=(2,4))
        
        tk.Label(fb, text="Output Folder:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 8)).pack(anchor="w")
        tk.Entry(fb, textvariable=self.batch_out, bg=C["border"], fg=C["text"], relief="flat", font=("Helvetica", 8)).pack(fill="x")
        self._btn(fb, "Select Output...", lambda: self.batch_out.set(filedialog.askdirectory(title="Output Folder") or self.batch_out.get())).pack(fill="x", pady=(2,4))
        
        self._batch_btn = tk.Button(fb, text="🚀 Auto-Run Batch", command=self._run_batch_async,
                                  bg=C["warn"], fg="#ffffff", font=("Segoe UI", 10, "bold"),
                                  relief="flat", pady=4, cursor="hand2")
        self._batch_btn.pack(fill="x", pady=(8,0))

        # -> Space spacer
        tk.Frame(sidebar, bg=C["panel"]).pack(fill="both", expand=True)

        # -> Save Result
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
        
        img_copy = self._rgb.copy()
        
        if v == "Original":
            self._current_view_rgb = img_copy
        elif v == "Bounding Boxes":
            for i in indices:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                label = f"{names[cls]} {conf:.2f}"
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 50, 50), 2)
                cv2.putText(img_copy, label, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self._current_view_rgb = img_copy
        elif v == "Mask Overlay":
            if masks is not None:
                # YOLO segmentation mask
                for i in indices:
                    mask = masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (img_copy.shape[1], img_copy.shape[0]))
                    img_copy[mask > 0.5] = img_copy[mask > 0.5] * 0.5 + np.array([255, 50, 50]) * 0.5
            else:
                # fallback if not a segmentation model
                self._current_view_rgb = img_copy # Just fallback
                
            for i in indices:
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 50, 50), 2)
                
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
