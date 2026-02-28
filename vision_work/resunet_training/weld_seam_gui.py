"""
Weld Seam Detection — Premium GUI Tester v2
Modern Tkinter interface with sidebar controls, live threshold, view toggles,
animated scan progress, and a detailed stats panel.

Run: python3 weld_seam_gui.py
Deps: torch, torchvision, opencv-python, scikit-image, Pillow, matplotlib
"""
import os, sys, threading, time, math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not found.  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# ─────────────────────────── colour palette ──────────────────────────────────
C = dict(
    bg       = "#0d1117",   # almost-black background
    panel    = "#161b22",   # sidebar / card bg
    border   = "#21262d",   # subtle borders
    acc1     = "#58a6ff",   # blue accent (buttons, focus)
    acc2     = "#3fb950",   # green (success / seam colour)
    acc3     = "#f78166",   # coral (overlay colour)
    warn     = "#d29922",   # amber warning
    txt      = "#c9d1d9",   # primary text
    txt2     = "#8b949e",   # secondary/muted text
    white    = "#ffffff",
    btn_bg   = "#21262d",
    btn_act  = "#388bfd",
)

FONT_H1  = ("Segoe UI", 16, "bold")
FONT_H2  = ("Segoe UI", 12, "bold")
FONT_BODY= ("Segoe UI", 10)
FONT_MONO= ("Courier New", 9)

# ═══════════════════════════ ResUNet (same as training) ══════════════════════

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.conv(x) + self.skip(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res = ResBlock(in_c, out_c); self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        s = self.res(x); return s, self.pool(s)

class Bridge(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__(); self.res = ResBlock(in_c, out_c)
    def forward(self, x): return self.res(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, 2)
        self.res = ResBlock(out_c * 2, out_c)
    def forward(self, x, skip):
        return self.res(torch.cat([self.up(x), skip], 1))

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
        s1,x = self.e1(x); s2,x = self.e2(x)
        s3,x = self.e3(x); s4,x = self.e4(x)
        x = self.bridge(x)
        x = self.d4(x,s4); x = self.d3(x,s3)
        x = self.d2(x,s2); x = self.d1(x,s1)
        return self.head(x)

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# ═══════════════════════════ Inference helpers ════════════════════════════════

def infer(img_bgr, model, device, threshold=0.5):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]
    inp     = cv2.resize(img_rgb, (512, 512))
    inp     = (inp / 255.0 - MEAN) / STD
    inp     = torch.tensor(inp).float().permute(2,0,1).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    prob_full = cv2.resize(prob, (w, h))
    return img_rgb, mask, prob_full

def build_overlay(img_rgb, mask, colour=(220,60,60), alpha=0.55):
    out = img_rgb.copy().astype(np.float32)
    m   = mask == 1
    out[m] = (1-alpha)*out[m] + alpha*np.array(colour)
    return np.clip(out, 0, 255).astype(np.uint8)

def build_skeleton(img_rgb, mask):
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask)
        out  = img_rgb.copy()
        out[skel] = [0, 230, 60]
        return out
    except ImportError:
        return build_overlay(img_rgb, mask, colour=(0,230,60))

def build_heatmap(prob_full, img_rgb):
    norm = (prob_full * 255).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb, 0.4, heat, 0.6, 0)
    return blended

def pil_from_np(arr):
    return Image.fromarray(arr.astype(np.uint8))

# ═══════════════════════════ Reusable widget helpers ═════════════════════════

def _btn(parent, text, cmd, bg=None, fg=None, font=FONT_BODY, pad=(10,6), **kw):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg or C["btn_bg"], fg=fg or C["txt"],
                  activebackground=C["acc1"], activeforeground=C["white"],
                  relief="flat", font=font, padx=pad[0], pady=pad[1],
                  cursor="hand2", borderwidth=0, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=C["border"]))
    b.bind("<Leave>", lambda e: b.config(bg=bg or C["btn_bg"]))
    return b

def _label(parent, text, font=FONT_BODY, fg=None, bg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg or C["txt"], bg=bg or C["bg"], **kw)

def _sep(parent, orient="h"):
    f = tk.Frame(parent, bg=C["border"],
                 height=1 if orient=="h" else 100,
                 width=100 if orient=="v" else 1)
    return f

# ═══════════════════════════ Main Application ═════════════════════════════════

class App(tk.Tk):
    VIEW_MODES = ["Overlay", "Mask", "Skeleton", "Heat Map"]

    def __init__(self):
        super().__init__()
        self.title("Weld Seam Detector  ·  ResUNet")
        self.geometry("1300x800")
        self.minsize(900, 600)
        self.configure(bg=C["bg"])

        # State
        self.model       = None
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path  = tk.StringVar(value=self._default_model())
        self.image_path  = tk.StringVar()
        self.threshold   = tk.DoubleVar(value=0.5)
        self.view_mode   = tk.StringVar(value="Overlay")

        self._orig_rgb   = None   # np array
        self._cur_mask   = None
        self._cur_prob   = None
        self._result_pil = None   # PIL image shown in canvas
        self._scan_angle = 0.0
        self._scanning   = False

        self._build()
        self._load_model_async()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _default_model(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "best_resunet_seam.pth")

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build(self):
        self._build_sidebar()
        self._build_main()
        self._build_statusbar()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = tk.Frame(self, bg=C["panel"], width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo / title
        tk.Frame(self.sidebar, bg=C["acc1"], height=3).pack(fill="x")
        hdr = tk.Frame(self.sidebar, bg=C["panel"], pady=16)
        hdr.pack(fill="x", padx=16)
        _label(hdr, "🔬 WeldVision", font=("Segoe UI", 14, "bold"),
               fg=C["white"], bg=C["panel"]).pack(anchor="w")
        _label(hdr, "ResUNet Seam Detector", font=("Segoe UI", 9),
               fg=C["txt2"], bg=C["panel"]).pack(anchor="w")

        _sep(self.sidebar).pack(fill="x", pady=4)

        # Model section
        self._section(self.sidebar, "Model Weights")
        mf = tk.Frame(self.sidebar, bg=C["panel"])
        mf.pack(fill="x", padx=12, pady=4)
        self._model_entry = tk.Entry(mf, textvariable=self.model_path,
                                     bg=C["border"], fg=C["txt"],
                                     insertbackground=C["txt"], relief="flat",
                                     font=FONT_MONO)
        self._model_entry.pack(fill="x", ipady=4)
        _btn(mf, "Browse .pth…", self._browse_model).pack(fill="x", pady=(4,0))

        self.model_status = _label(self.sidebar, "⏳ Loading…",
                                   font=("Segoe UI", 9), fg=C["warn"],
                                   bg=C["panel"])
        self.model_status.pack(anchor="w", padx=12, pady=(2,8))

        _sep(self.sidebar).pack(fill="x", pady=4)

        # Image section
        self._section(self.sidebar, "Test Image")
        img_frame = tk.Frame(self.sidebar, bg=C["panel"])
        img_frame.pack(fill="x", padx=12, pady=4)
        self._img_entry = tk.Entry(img_frame, textvariable=self.image_path,
                                   bg=C["border"], fg=C["txt"],
                                   insertbackground=C["txt"], relief="flat",
                                   font=FONT_MONO)
        self._img_entry.pack(fill="x", ipady=4)
        _btn(img_frame, "Browse Image…", self._browse_image).pack(fill="x", pady=(4,0))

        # Tiny preview
        self._thumb_canvas = tk.Canvas(self.sidebar, bg=C["border"],
                                       width=230, height=120,
                                       highlightthickness=0)
        self._thumb_canvas.pack(padx=12, pady=6)
        self._thumb_canvas.create_text(115, 60, text="No image loaded",
                                       fill=C["txt2"], font=FONT_BODY,
                                       tags="placeholder")

        _sep(self.sidebar).pack(fill="x", pady=4)

        # Threshold
        self._section(self.sidebar, "Detection Threshold")
        tf = tk.Frame(self.sidebar, bg=C["panel"])
        tf.pack(fill="x", padx=12, pady=4)
        self._thresh_lbl = _label(tf, f"0.50", font=("Segoe UI", 11, "bold"),
                                  fg=C["acc1"], bg=C["panel"])
        self._thresh_lbl.pack(anchor="e")
        sl = tk.Scale(tf, from_=0.01, to=0.99, resolution=0.01,
                      variable=self.threshold, orient="horizontal",
                      length=230, bg=C["panel"], fg=C["txt"],
                      troughcolor=C["border"], sliderrelief="flat",
                      activebackground=C["acc1"], highlightthickness=0,
                      showvalue=False, command=self._on_threshold_change)
        sl.pack(fill="x")
        _label(tf, "Lower = detect more  |  Higher = stricter",
               font=("Segoe UI", 8), fg=C["txt2"], bg=C["panel"]).pack()

        _sep(self.sidebar).pack(fill="x", pady=4)

        # View mode
        self._section(self.sidebar, "View Mode")
        vf = tk.Frame(self.sidebar, bg=C["panel"])
        vf.pack(fill="x", padx=12, pady=4)
        self._mode_btns = {}
        for mode in self.VIEW_MODES:
            b = tk.Button(vf, text=mode, font=FONT_BODY, relief="flat",
                          bg=C["btn_bg"], fg=C["txt"], padx=6, pady=5,
                          cursor="hand2", borderwidth=0,
                          command=lambda m=mode: self._set_view(m))
            b.pack(fill="x", pady=2)
            self._mode_btns[mode] = b
        self._set_view("Overlay")

        _sep(self.sidebar).pack(fill="x", pady=4)

        # Action buttons
        self._run_btn = _btn(self.sidebar, "▶  Run Inference",
                             self._run_async,
                             bg=C["acc1"], fg=C["white"],
                             font=("Segoe UI", 11, "bold"), pad=(12, 8))
        self._run_btn.pack(fill="x", padx=12, pady=4)
        self._run_btn.bind("<Enter>", lambda e: self._run_btn.config(bg="#79c0ff"))
        self._run_btn.bind("<Leave>", lambda e: self._run_btn.config(bg=C["acc1"]))

        _btn(self.sidebar, "💾  Save Result", self._save_result).pack(
            fill="x", padx=12, pady=2)

    # ── Main canvas area ──────────────────────────────────────────────────────
    def _build_main(self):
        self.main_frame = tk.Frame(self, bg=C["bg"])
        self.main_frame.pack(side="left", fill="both", expand=True)

        # Top bar
        top = tk.Frame(self.main_frame, bg=C["panel"], height=40)
        top.pack(fill="x")
        top.pack_propagate(False)
        self._view_lbl = _label(top, "Overlay  ·  awaiting image",
                                font=("Segoe UI", 10), fg=C["txt2"],
                                bg=C["panel"])
        self._view_lbl.pack(side="left", padx=16, pady=10)

        self._device_lbl = _label(top,
                                  f"Device: {self.device.type.upper()}",
                                  font=("Segoe UI", 9), fg=C["acc2"],
                                  bg=C["panel"])
        self._device_lbl.pack(side="right", padx=16)

        # Canvas
        self.canvas = tk.Canvas(self.main_frame, bg=C["bg"],
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(
            500, 300,
            text="Browse an image and click  ▶ Run Inference",
            fill=C["txt2"], font=("Segoe UI", 14), tags="welcome")
        self._canvas_img_ref = None
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Stats bar
        stats_bar = tk.Frame(self.main_frame, bg=C["panel"], height=36)
        stats_bar.pack(fill="x", side="bottom")
        stats_bar.pack_propagate(False)
        self._stats_lbl = _label(stats_bar, "", font=("Segoe UI", 9),
                                 fg=C["txt2"], bg=C["panel"])
        self._stats_lbl.pack(side="left", padx=16, pady=8)

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        pass  # embedded in sidebar

    # ── Section header helper ─────────────────────────────────────────────────
    def _section(self, parent, text):
        f = tk.Frame(parent, bg=C["panel"])
        f.pack(fill="x", padx=12, pady=(10, 2))
        tk.Frame(f, bg=C["acc1"], width=3).pack(side="left", fill="y")
        _label(f, f"  {text}", font=("Segoe UI", 9, "bold"),
               fg=C["txt2"], bg=C["panel"]).pack(side="left")

    # ── Browse callbacks ──────────────────────────────────────────────────────
    def _browse_model(self):
        p = filedialog.askopenfilename(title="Select model weights",
            filetypes=[("PyTorch weights", "*.pth"), ("All files", "*.*")])
        if p:
            self.model_path.set(p)
            self._load_model_async()

    def _browse_image(self):
        p = filedialog.askopenfilename(title="Select test image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")])
        if p:
            self.image_path.set(p)
            self._load_preview(p)

    # ── Image preview thumbnail ───────────────────────────────────────────────
    def _load_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((230, 120), Image.LANCZOS)
            self._thumb_ref = ImageTk.PhotoImage(img)
            self._thumb_canvas.delete("all")
            cx, cy = 115, 60
            self._thumb_canvas.create_image(cx, cy, anchor="center",
                                            image=self._thumb_ref)
        except Exception:
            pass

    # ── Threshold slider ──────────────────────────────────────────────────────
    def _on_threshold_change(self, val):
        v = float(val)
        self._thresh_lbl.config(text=f"{v:.2f}")
        # Live update if we already have results
        if self._orig_rgb is not None and self._cur_mask is not None:
            self._apply_view_async()

    # ── View mode buttons ─────────────────────────────────────────────────────
    def _set_view(self, mode):
        self.view_mode.set(mode)
        for m, btn in self._mode_btns.items():
            if m == mode:
                btn.config(bg=C["acc1"], fg=C["white"])
            else:
                btn.config(bg=C["btn_bg"], fg=C["txt"])
        if self._orig_rgb is not None and self._cur_mask is not None:
            self._apply_view_async()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model_async(self):
        self._set_model_status("⏳ Loading…", C["warn"])
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        path = self.model_path.get()
        if not os.path.exists(path):
            self.after(0, self._set_model_status, f"✘ Not found", C["acc3"])
            return
        try:
            m = ResUNet().to(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.eval()
            self.model = m
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            self.after(0, self._set_model_status,
                       f"✔ Ready  ({params/1e6:.1f}M params)", C["acc2"])
        except Exception as e:
            self.after(0, self._set_model_status, f"✘ {e}", C["acc3"])

    def _set_model_status(self, msg, color):
        self.model_status.config(text=msg, fg=color)

    # ── Scan animation ────────────────────────────────────────────────────────
    def _start_scan(self):
        self._scanning = True
        self._scan_angle = 0.0
        self._animate_scan()

    def _stop_scan(self):
        self._scanning = False

    def _animate_scan(self):
        if not self._scanning:
            return
        self._scan_angle = (self._scan_angle + 4) % 360
        # Draw a spinning arc over the welcome text area
        w = self.canvas.winfo_width() or 900
        h = self.canvas.winfo_height() or 550
        cx, cy, r = w//2, h//2, 40
        self.canvas.delete("spinner")
        start = self._scan_angle
        for i in range(8):
            a = math.radians(start + i * 45)
            x0 = cx + (r-8)*math.cos(a)
            y0 = cy + (r-8)*math.sin(a)
            x1 = cx + r*math.cos(a)
            y1 = cy + r*math.sin(a)
            alpha = int(255 * (i+1)/8)
            col   = f"#{alpha:02x}{alpha:02x}ff"[:7]  # blue fade
            self.canvas.create_line(x0, y0, x1, y1, width=3,
                                    fill=C["acc1"], tags="spinner")
        self.after(40, self._animate_scan)

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_async(self):
        if not self.model:
            messagebox.showwarning("No model", "Model is still loading, please wait.")
            return
        path = self.image_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("No image", "Please select a valid image file.")
            return
        self._run_btn.config(state="disabled", text="⏳  Running…")
        self.canvas.delete("welcome")
        self._start_scan()
        threading.Thread(target=self._run_inference, args=(path,), daemon=True).start()

    def _run_inference(self, path):
        t0 = time.time()
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                raise ValueError("Cannot read image")
            threshold = self.threshold.get()
            img_rgb, mask, prob = infer(img_bgr, self.model, self.device, threshold)
            self._orig_rgb  = img_rgb
            self._cur_mask  = mask
            self._cur_prob  = prob
            elapsed = time.time() - t0
            self.after(0, self._inference_done, elapsed)
        except Exception as e:
            self.after(0, self._inference_error, str(e))

    def _inference_done(self, elapsed):
        self._stop_scan()
        self.canvas.delete("spinner")
        mask   = self._cur_mask
        h, w   = mask.shape
        pixels = int(mask.sum())
        coverage = 100 * pixels / (h * w)
        self._stats_lbl.config(
            text=f"  Seam pixels: {pixels:,}   Coverage: {coverage:.3f}%   "
                 f"Image: {w}×{h}   Inference: {elapsed*1000:.0f} ms   "
                 f"Threshold: {self.threshold.get():.2f}")
        self._apply_view()
        self._run_btn.config(state="normal", text="▶  Run Inference")

    def _inference_error(self, msg):
        self._stop_scan()
        self.canvas.delete("spinner")
        messagebox.showerror("Inference error", msg)
        self._run_btn.config(state="normal", text="▶  Run Inference")

    # ── View rendering ────────────────────────────────────────────────────────
    def _apply_view_async(self):
        threading.Thread(target=self._apply_view, daemon=True).start()

    def _apply_view(self):
        mode = self.view_mode.get()
        thr  = self.threshold.get()

        # Recompute mask from prob with new threshold if needed
        if self._cur_prob is not None:
            mask = (self._cur_prob > thr).astype(np.uint8)
            self._cur_mask = mask
        else:
            mask = self._cur_mask

        img_rgb = self._orig_rgb
        if   mode == "Overlay":    vis = build_overlay(img_rgb, mask)
        elif mode == "Mask":       vis = (mask * 255)[..., np.newaxis].repeat(3, axis=2)
        elif mode == "Skeleton":   vis = build_skeleton(img_rgb, mask)
        elif mode == "Heat Map":   vis = build_heatmap(self._cur_prob, img_rgb)
        else:                      vis = img_rgb

        pil = pil_from_np(vis)
        self._result_pil = pil
        self.after(0, self._show_image, pil, mode)

    def _show_image(self, pil, mode):
        cw = self.canvas.winfo_width()  or 1000
        ch = self.canvas.winfo_height() or 600
        ratio  = min(cw / pil.width, ch / pil.height, 1.0)
        new_w  = int(pil.width  * ratio)
        new_h  = int(pil.height * ratio)
        resized = pil.resize((new_w, new_h), Image.LANCZOS)
        self._canvas_img_ref = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor="center",
                                 image=self._canvas_img_ref, tags="result")
        # mode label overlay  
        self._view_lbl.config(text=f"{mode}  ·  {os.path.basename(self.image_path.get())}")

    def _on_canvas_resize(self, event):
        if self._result_pil:
            self._show_image(self._result_pil, self.view_mode.get())

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save_result(self):
        if not self._result_pil:
            messagebox.showinfo("Nothing to save", "Run inference first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"seam_{self.view_mode.get().lower().replace(' ','_')}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")])
        if path:
            self._result_pil.save(path)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
