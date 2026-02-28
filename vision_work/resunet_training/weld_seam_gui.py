"""
WeldVision GUI — V1 Sidebar Style with V5 Controls
==================================================
- Classic single-pane dark sidebar layout
- Standard OS file dialog for picking an image
- Dropdown/Radio for View Mode (Original | Overlay | Mask | Heat Map | Skeleton)
- Live threshold slider updates mask without re-running
- Saves the currently selected view exactly as seen on screen

Run: python3 weld_seam_gui.py
"""
import os, sys, threading, time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# ─── ResUNet (exact match to training weights) ────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c))
        self.skip = nn.Sequential() if (stride == 1 and in_c == out_c) else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.conv(x) + self.skip(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__(); self.res = ResBlock(in_c, out_c); self.pool = nn.MaxPool2d(2)
    def forward(self, x): s = self.res(x); return s, self.pool(s)

class Bridge(nn.Module):
    def __init__(self, in_c, out_c): super().__init__(); self.res = ResBlock(in_c, out_c)
    def forward(self, x): return self.res(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, 2); self.res = ResBlock(out_c * 2, out_c)
    def forward(self, x, skip): return self.res(torch.cat([self.up(x), skip], 1))

class ResUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.e1 = EncoderBlock(in_c, 64);   self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256);   self.e4 = EncoderBlock(256, 512)
        self.bridge = Bridge(512, 1024)
        self.d4 = DecoderBlock(1024, 512);  self.d3 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128);   self.d1 = DecoderBlock(128, 64)
        self.head = nn.Conv2d(64, out_c, 1)
    def forward(self, x):
        s1,x=self.e1(x); s2,x=self.e2(x); s3,x=self.e3(x); s4,x=self.e4(x)
        x=self.bridge(x); x=self.d4(x,s4); x=self.d3(x,s3); x=self.d2(x,s2); x=self.d1(x,s1)
        return self.head(x)

# ─── Image processing helpers ─────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def run_inference(img_bgr, model, device, threshold):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    resized = cv2.resize(img_rgb, (512, 512))
    normed = (resized / 255.0 - MEAN) / STD
    tensor = torch.tensor(normed).float().permute(2, 0, 1).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    prob_full = cv2.resize(prob, (w, h))
    return img_rgb, mask, prob_full

def make_overlay(img_rgb, mask):
    overlay = img_rgb.copy()
    overlay[mask == 1] = [255, 50, 50]   # same as Colab
    return overlay

def make_skeleton(img_rgb, mask):
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask)
        skel_vis = img_rgb.copy()
        skel_vis[skeleton] = [0, 255, 0]  # same as Colab
        return skel_vis
    except ImportError:
        return make_overlay(img_rgb, mask)

def make_mask(mask):
    return np.stack([mask * 255] * 3, axis=2).astype(np.uint8)

def make_heatmap(prob, img_rgb):
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.35, heat, 0.65, 0)

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
        self.title("WeldVision · Advanced Inference")
        self.geometry("1100x700")
        self.minsize(900, 600)
        self.configure(bg=C["bg"])

        # State
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self.image_path = tk.StringVar(value="")
        self.model_path = tk.StringVar(value=self._default_model())
        self.threshold  = tk.DoubleVar(value=0.5)
        self.view_mode  = tk.StringVar(value="Overlay")

        # Data
        self._rgb  = None
        self._mask = None
        self._prob = None
        self._current_view_rgb = None
        self._pil_img = None
        self._tk_img  = None

        self._build()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _default_model(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "best_resunet_seam.pth")

    # ─── UI Building ──────────────────────────────────────────────────────────
    def _build(self):
        # Top Header
        hdr = tk.Frame(self, bg=C["panel"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text=" 🔬 WeldVision", font=("Segoe UI", 14, "bold"),
                 bg=C["panel"], fg=C["text"]).pack(side="left", padx=10)
        self._model_status = tk.Label(hdr, text="Loading model...", font=("Segoe UI", 10),
                                      bg=C["panel"], fg=C["warn"])
        self._model_status.pack(side="right", padx=15)
        tk.Label(hdr, text=f"Device: {self.device.type.upper()}", font=("Segoe UI", 10),
                 bg=C["panel"], fg=C["green"]).pack(side="right", padx=10)

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

        # -> Inference Action
        self._section(sidebar, "Inference")
        self._run_btn = tk.Button(sidebar, text="▶ Run Detection", command=self._run_async,
                                  bg=C["accent"], fg="#ffffff", font=("Segoe UI", 11, "bold"),
                                  relief="flat", pady=6, cursor="hand2")
        self._run_btn.pack(fill="x", padx=12, pady=8)

        # -> Controls
        self._section(sidebar, "Threshold & View")
        f2 = tk.Frame(sidebar, bg=C["panel"])
        f2.pack(fill="x", padx=12, pady=2)
        tk.Label(f2, text="Mask Threshold:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 9)).pack(anchor="w")
        self._thr_lbl = tk.Label(f2, text="str(0.50)", bg=C["panel"], fg=C["accent"], font=("Segoe UI", 10, "bold"))
        self._thr_lbl.pack(anchor="e", pady=(0, 2))
        self._thr_lbl.config(text="0.50")
        
        tk.Scale(f2, from_=0.01, to=0.99, resolution=0.01, variable=self.threshold,
                 orient="horizontal", bg=C["panel"], fg=C["text"], troughcolor=C["border"],
                 highlightthickness=0, showvalue=False, command=self._on_thr_change).pack(fill="x")

        # View Radios
        tk.Label(f2, text="View Mode:", bg=C["panel"], fg=C["text2"], font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 2))
        for mode in ["Original", "Overlay", "Mask", "Heat Map", "Skeleton"]:
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

    def _browse_model(self):
        p = filedialog.askopenfilename(title="Select model", filetypes=[("PTH", "*.pth"), ("All", "*.*")])
        if p:
            self.model_path.set(p)
            threading.Thread(target=self._load_model, daemon=True).start()

    def _on_thr_change(self, val):
        self._thr_lbl.config(text=f"{float(val):.2f}")
        val_f = float(val)
        if self._prob is not None and self._rgb is not None:
            # Recompute mask and view instantly
            self._mask = (self._prob > val_f).astype(np.uint8)
            self._on_view_change()

    def _on_view_change(self):
        if self._rgb is None: return
        v = self.view_mode.get()
        if v == "Original":
            self._current_view_rgb = self._rgb
        elif v == "Overlay":
            self._current_view_rgb = make_overlay(self._rgb, self._mask)
        elif v == "Mask":
            self._current_view_rgb = make_mask(self._mask)
        elif v == "Heat Map":
            self._current_view_rgb = make_heatmap(self._prob, self._rgb)
        elif v == "Skeleton":
            self._current_view_rgb = make_skeleton(self._rgb, self._mask)
        
        # Stats update
        px = int(self._mask.sum())
        h, w = self._mask.shape
        self._lbl_stats.config(text=f"Mode: {v}  |  Seam: {px:,} px  |  Coverage: {100*px/(h*w):.3f}%")

        # Redraw
        self._pil_img = Image.fromarray(self._current_view_rgb)
        self._redraw()

    def _save_result(self):
        if self._rgb is None:
            messagebox.showinfo("Wait", "Run inference first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         initialfile=f"result_{os.path.basename(self.image_path.get())}.png",
                                         filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if p:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            ov = make_overlay(self._rgb, self._mask)
            sk = make_skeleton(self._rgb, self._mask)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(self._rgb); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(ov);        axes[1].set_title("Seam Mask Overlay"); axes[1].axis("off")
            axes[2].imshow(sk);        axes[2].set_title("Centerline (Green)"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(p, dpi=150)
            plt.close(fig)
            messagebox.showinfo("Saved", f"3-panel Colab-style view saved to:\n{p}")


    # ─── Async Logic ──────────────────────────────────────────────────────────
    def _load_model(self):
        p = self.model_path.get()
        if not os.path.exists(p):
            self.after(0, lambda: self._model_status.config(text="Model File Missing", fg=C["err"]))
            return
        self.after(0, lambda: self._model_status.config(text="Loading Model...", fg=C["warn"]))
        try:
            m = ResUNet().to(self.device)
            m.load_state_dict(torch.load(p, map_location=self.device))
            m.eval()
            self.model = m
            self.after(0, lambda: self._model_status.config(text="Model Ready", fg=C["green"]))
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
                
                rgb, mask, prob = run_inference(bgr, self.model, self.device, self.threshold.get())
                self._rgb, self._mask, self._prob = rgb, mask, prob
                
                ms = (time.time() - t0) * 1000
                self.after(0, self._lbl_title.config, {"text": f"{os.path.basename(img_path)} ({ms:.1f}ms)"})
                self.after(0, self._on_view_change)
            except Exception as e:
                self.after(0, messagebox.showerror, "Error", str(e))
                self.after(0, self._lbl_title.config, {"text": "Error running inference"})
            finally:
                self.after(0, self._run_btn.config, {"state": "normal", "text": "▶ Run Detection"})
        
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
