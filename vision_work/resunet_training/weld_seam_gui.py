"""
WeldVision GUI — Simple Colab-style output
==========================================
- OS native file picker (browse anywhere on the computer)
- 3 panels: Original | Middle (interactive) | Skeleton
- Bottom toggle: Overlay  Mask  Heat Map  Skeleton
- Threshold slider (live update without re-running)
- Save button exports 3-panel composite

Run: python3 weld_seam_gui.py
"""
import os, sys, threading, time
import tkinter as tk
from tkinter import filedialog, messagebox
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

def np_to_pil(arr): return Image.fromarray(arr.astype(np.uint8))

# ─── GUI ──────────────────────────────────────────────────────────────────────
VIEWS = ["Overlay", "Mask", "Heat Map", "Skeleton"]

BG      = "#1c2128"
PANEL   = "#22272e"
BORDER  = "#444c56"
WHITE   = "#cdd9e5"
GREY    = "#768390"
BLUE    = "#388bfd"
GREEN   = "#3fb950"
RED     = "#ff7b72"
AMBER   = "#d29922"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WeldVision  ·  ResUNet Seam Detector")
        self.geometry("1200x680")
        self.minsize(900, 560)
        self.configure(bg=BG)

        # State
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = None
        self._view     = tk.StringVar(value="Overlay")
        self._thr      = tk.DoubleVar(value=0.5)

        self._img_path = None
        self._rgb      = None
        self._mask     = None
        self._prob     = None

        # PIL image refs for the 3 panels
        self._pil      = {"orig": None, "mid": None, "skel": None}
        self._tkimg    = {}

        self._build()
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── Find default model path ───────────────────────────────────────────────
    def _default_model(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "best_resunet_seam.pth")

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build(self):
        # ── Top toolbar ───────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg=PANEL, height=50)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)
        tk.Frame(toolbar, bg=BLUE, width=4).pack(side="left", fill="y")
        tk.Label(toolbar, text="  🔬 WeldVision", font=("Segoe UI", 13, "bold"),
                 bg=PANEL, fg=WHITE).pack(side="left", padx=6)

        # Buttons
        for text, cmd, color in [
            ("📂  Open Image",  self._open,   BLUE),
            ("▶   Run",         self._run,    GREEN),
            ("💾  Save Result",  self._save,   "#444c56"),
        ]:
            b = tk.Button(toolbar, text=text, command=cmd, bg=color, fg=WHITE,
                          relief="flat", font=("Segoe UI", 10, "bold"),
                          padx=14, pady=6, cursor="hand2", borderwidth=0,
                          activebackground="#ffffff22", activeforeground=WHITE)
            b.pack(side="left", padx=5, pady=8)

        # Status labels (right side)
        self._status_lbl = tk.Label(toolbar, text="Open an image to begin",
                                    font=("Segoe UI", 9), bg=PANEL, fg=GREY)
        self._status_lbl.pack(side="right", padx=16)
        self._model_lbl = tk.Label(toolbar, text=f"⏳ Loading…  {self.device.type.upper()}",
                                   font=("Segoe UI", 9), bg=PANEL, fg=AMBER)
        self._model_lbl.pack(side="right", padx=8)

        # ── 3-Panel viewer ────────────────────────────────────────────────────
        viewer = tk.Frame(self, bg=BG)
        viewer.pack(fill="both", expand=True, padx=10, pady=6)

        panel_titles = [("Original", "orig"), ("Detection View", "mid"),
                        ("Skeleton / Centerline", "skel")]
        self._canvases = {}
        for title, key in panel_titles:
            col = tk.Frame(viewer, bg=PANEL, bd=0)
            col.pack(side="left", fill="both", expand=True, padx=4)
            # Header
            hdr = tk.Frame(col, bg="#2d333b", height=30)
            hdr.pack(fill="x"); hdr.pack_propagate(False)
            tk.Label(hdr, text=title, font=("Segoe UI", 10, "bold"),
                     bg="#2d333b", fg=WHITE).pack(side="left", padx=10, pady=5)
            # Canvas
            c = tk.Canvas(col, bg=PANEL, highlightthickness=0)
            c.pack(fill="both", expand=True)
            c.create_text(10, 20, anchor="nw", text="—", fill=GREY,
                          font=("Segoe UI", 10), tags="placeholder")
            self._canvases[key] = c
            c.bind("<Configure>", lambda e, k=key: self._redraw(k))

        # ── Bottom control bar ────────────────────────────────────────────────
        bar = tk.Frame(self, bg=PANEL, height=52)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x")

        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(side="left", padx=14, pady=8)

        tk.Label(inner, text="Middle panel:", font=("Segoe UI", 9),
                 bg=PANEL, fg=GREY).pack(side="left", padx=(0, 8))

        self._view_btns = {}
        for v in VIEWS:
            b = tk.Button(inner, text=v, command=lambda mv=v: self._set_view(mv),
                          bg=BORDER, fg=WHITE, relief="flat", font=("Segoe UI", 9),
                          padx=12, pady=4, cursor="hand2", borderwidth=0)
            b.pack(side="left", padx=3)
            self._view_btns[v] = b
        self._set_view("Overlay")

        # Threshold slider
        tk.Label(inner, text="  Threshold:", font=("Segoe UI", 9),
                 bg=PANEL, fg=GREY).pack(side="left", padx=(16, 4))
        self._thr_lbl = tk.Label(inner, text="0.50", font=("Segoe UI", 9, "bold"),
                                  bg=PANEL, fg=BLUE)
        self._thr_lbl.pack(side="left", padx=(0, 4))
        tk.Scale(inner, from_=0.01, to=0.99, resolution=0.01, variable=self._thr,
                 orient="horizontal", length=140, bg=PANEL, fg=WHITE, troughcolor=BORDER,
                 sliderrelief="flat", activebackground=BLUE, highlightthickness=0,
                 showvalue=False, command=self._on_thr).pack(side="left")

        # Stats
        self._stats_lbl = tk.Label(bar, text="", font=("Segoe UI", 9), bg=PANEL, fg=GREY)
        self._stats_lbl.pack(side="right", padx=16)

        # Keyboard shortcuts
        self.bind("<Return>", lambda e: self._run())
        self.bind("o",        lambda e: self._open())
        self.bind("s",        lambda e: self._save())
        self.bind("t",        lambda e: self._cycle_view())

    # ── Open image ────────────────────────────────────────────────────────────
    def _open(self):
        path = filedialog.askopenfilename(
            title="Select a test image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")])
        if path:
            self._img_path = path
            self._status(f"Loaded: {os.path.basename(path)}")
            if self.model:
                self._run()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        path = self._default_model()
        if not os.path.exists(path):
            self.after(0, lambda: self._model_lbl.config(
                text="✘ Model not found — place best_resunet_seam.pth here", fg=RED))
            return
        try:
            m = ResUNet().to(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.eval()
            self.model = m
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            self.after(0, lambda: self._model_lbl.config(
                text=f"✔ Model ready · {n/1e6:.1f}M params · {self.device.type.upper()}",
                fg=GREEN))
        except Exception as ex:
            self.after(0, lambda: self._model_lbl.config(text=f"✘ {ex}", fg=RED))

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run(self):
        if not self.model:
            messagebox.showwarning("Not ready", "Model is still loading, please wait.")
            return
        if not self._img_path or not os.path.exists(self._img_path):
            messagebox.showwarning("No image", "Open an image first (📂 button or press O).")
            return
        self._status("⏳ Running inference…")
        threading.Thread(target=self._infer, daemon=True).start()

    def _infer(self):
        t0 = time.time()
        try:
            bgr = cv2.imread(self._img_path)
            if bgr is None:
                raise ValueError("Cannot read this file as an image.")
            rgb, mask, prob = run_inference(bgr, self.model, self.device, self._thr.get())
            self._rgb = rgb; self._mask = mask; self._prob = prob
            ms = (time.time() - t0) * 1000
            self.after(0, self._display, ms)
        except Exception as ex:
            self.after(0, self._status, f"✘ Error: {ex}")

    def _display(self, ms):
        rgb = self._rgb; mask = self._mask; prob = self._prob
        h, w = mask.shape
        px   = int(mask.sum())

        self._pil["orig"] = np_to_pil(rgb)
        self._pil["mid"]  = np_to_pil(self._build_mid(rgb, mask, prob))
        self._pil["skel"] = np_to_pil(make_skeleton(rgb, mask))

        for k in ("orig", "mid", "skel"):
            self._redraw(k)

        try:
            from skimage.morphology import skeletonize
            length = int(skeletonize(mask).sum())
        except Exception:
            length = px

        self._stats_lbl.config(
            text=f"  {os.path.basename(self._img_path)}   |"
                 f"  Seam: {px:,} px  "
                 f"  Coverage: {100*px/(h*w):.3f}%  "
                 f"  Length: ~{length:,} px  "
                 f"  Time: {ms:.0f} ms   Threshold: {self._thr.get():.2f}  ")
        self._status(f"✔ Done — {px:,} seam pixels detected in {ms:.0f} ms")

    # ── View controls ─────────────────────────────────────────────────────────
    def _build_mid(self, rgb, mask, prob):
        v = self._view.get()
        if   v == "Overlay":  return make_overlay(rgb, mask)
        elif v == "Mask":     return make_mask(mask)
        elif v == "Heat Map": return make_heatmap(prob, rgb)
        elif v == "Skeleton": return make_skeleton(rgb, mask)
        return make_overlay(rgb, mask)

    def _set_view(self, mode):
        self._view.set(mode)
        for v, b in self._view_btns.items():
            b.config(bg=BLUE if v == mode else BORDER,
                     fg=WHITE)
        if self._rgb is not None:
            self._pil["mid"] = np_to_pil(
                self._build_mid(self._rgb, self._mask, self._prob))
            self._redraw("mid")

    def _cycle_view(self):
        i = VIEWS.index(self._view.get())
        self._set_view(VIEWS[(i + 1) % len(VIEWS)])

    def _on_thr(self, val):
        self._thr_lbl.config(text=f"{float(val):.2f}")
        if self._prob is not None:
            # Recompute mask from probability without re-running the model
            self._mask = (self._prob > float(val)).astype(np.uint8)
            self._pil["mid"]  = np_to_pil(self._build_mid(self._rgb, self._mask, self._prob))
            self._pil["skel"] = np_to_pil(make_skeleton(self._rgb, self._mask))
            self._redraw("mid")
            self._redraw("skel")

    # ── Image rendering ───────────────────────────────────────────────────────
    def _redraw(self, key):
        pil = self._pil.get(key)
        if pil is None:
            return
        c  = self._canvases[key]
        cw = c.winfo_width()  or 380
        ch = c.winfo_height() or 400
        ratio  = min(cw / pil.width, ch / pil.height)
        nw, nh = int(pil.width * ratio), int(pil.height * ratio)
        resized = pil.resize((nw, nh), Image.LANCZOS)
        ref = ImageTk.PhotoImage(resized)
        self._tkimg[key] = ref          # keep alive
        c.delete("all")
        c.create_image(cw // 2, ch // 2, anchor="center", image=ref)

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self):
        if self._rgb is None:
            messagebox.showinfo("Nothing to save", "Run inference on an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"seam_{os.path.splitext(os.path.basename(self._img_path))[0]}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        ov = make_overlay(self._rgb, self._mask)
        sk = make_skeleton(self._rgb, self._mask)
        composite = np.concatenate([self._rgb, ov, sk], axis=1)
        cv2.imwrite(path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Saved", f"3-panel composite saved to:\n{path}")

    # ── Status bar ────────────────────────────────────────────────────────────
    def _status(self, msg):
        self._status_lbl.config(text=msg)


if __name__ == "__main__":
    App().mainloop()
