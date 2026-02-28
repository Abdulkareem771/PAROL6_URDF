"""
Weld Seam Detection - Local GUI Tester
Uses the trained ResUNet model (best_resunet_seam.pth) to detect weld seams.
Requires: torch, torchvision, opencv-python, matplotlib, scikit-image, Pillow
Run: python3 weld_seam_gui.py
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")  # No display needed — we render to buffer
import matplotlib.pyplot as plt

# ── Try importing torch (guide user if missing) ──────────────────────────────
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not found. Please install it: pip install torch torchvision")
    sys.exit(1)

# ── ResUNet Architecture (must match training script exactly) ─────────────────
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
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

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res  = ResBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res(x)
        return skip, self.pool(skip)


class Bridge(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res = ResBlock(in_c, out_c)

    def forward(self, x):
        return self.res(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up  = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.res = ResBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        return self.res(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.e1 = EncoderBlock(in_c, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.bridge = Bridge(512, 1024)
        self.d4 = DecoderBlock(1024, 512)
        self.d3 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d1 = DecoderBlock(128, 64)
        self.head = nn.Conv2d(64, out_c, kernel_size=1)

    def forward(self, x):
        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        s4, x = self.e4(x)
        x = self.bridge(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.head(x)


# ── Inference helper ──────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def run_inference(img_bgr, model, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]
    resized = cv2.resize(img_rgb, (512, 512))
    normed  = (resized / 255.0 - MEAN) / STD
    inp     = torch.tensor(normed).float().permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(inp))
        mask = (pred > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Red overlay
    overlay = img_rgb.copy()
    overlay[mask == 1] = [220, 30, 30]

    # Green skeleton centerline
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask)
        skel_vis = img_rgb.copy()
        skel_vis[skeleton] = [0, 230, 0]
    except ImportError:
        skeleton = None
        skel_vis = overlay

    return img_rgb, mask, overlay, skel_vis, int(mask.sum())


def build_result_image(img_rgb, mask, overlay, skel_vis):
    """Render a 3-panel matplotlib figure and return as PIL Image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1a1a2e")
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", color="white", fontsize=13, pad=8)

    axes[1].imshow(overlay)
    axes[1].set_title("Seam Mask Overlay (Red)", color="white", fontsize=13, pad=8)

    axes[2].imshow(skel_vis)
    axes[2].set_title("Centerline Path (Green)", color="white", fontsize=13, pad=8)

    plt.tight_layout(pad=1.5)

    # Render to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(buf)


# ── Main GUI Application ──────────────────────────────────────────────────────
class WeldSeamApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Weld Seam Detector — ResUNet")
        self.geometry("1100x720")
        self.resizable(True, True)
        self.configure(bg="#1a1a2e")

        self.model       = None
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_path  = tk.StringVar()
        self.model_path  = tk.StringVar(value=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "best_resunet_seam.pth"))
        self.result_img  = None
        self._build_ui()
        self._load_model_async()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        BG    = "#1a1a2e"
        PANEL = "#16213e"
        ACC   = "#e94560"
        TXT   = "#eaeaea"
        BTN   = "#0f3460"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel",  background=BG,    foreground=TXT, font=("Inter", 11))
        style.configure("TButton", background=BTN,   foreground=TXT, font=("Inter", 11, "bold"), padding=8)
        style.configure("TEntry",  fieldbackground=PANEL, foreground=TXT)
        style.configure("TFrame",  background=BG)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=ACC, height=52)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔥  Weld Seam Detection — ResUNet", bg=ACC, fg="white",
                 font=("Inter", 15, "bold")).pack(side="left", padx=16, pady=12)
        self.status_lbl = tk.Label(hdr, text="Loading model…", bg=ACC, fg="white",
                                   font=("Inter", 11, "italic"))
        self.status_lbl.pack(side="right", padx=16)

        # ── Control panel ─────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG, pady=12)
        ctrl.pack(fill="x", padx=20)

        # Model path row
        tk.Label(ctrl, text="Model (.pth):", bg=BG, fg=TXT,
                 font=("Inter", 10)).grid(row=0, column=0, sticky="w", pady=4)
        tk.Entry(ctrl, textvariable=self.model_path, width=55, bg=PANEL, fg=TXT,
                 insertbackground=TXT, relief="flat").grid(row=0, column=1, padx=8)
        tk.Button(ctrl, text="Browse…", command=self._browse_model,
                  bg=BTN, fg=TXT, relief="flat", font=("Inter", 10), cursor="hand2",
                  padx=8).grid(row=0, column=2)

        # Image path row
        tk.Label(ctrl, text="Test Image:", bg=BG, fg=TXT,
                 font=("Inter", 10)).grid(row=1, column=0, sticky="w", pady=4)
        tk.Entry(ctrl, textvariable=self.image_path, width=55, bg=PANEL, fg=TXT,
                 insertbackground=TXT, relief="flat").grid(row=1, column=1, padx=8)
        tk.Button(ctrl, text="Browse…", command=self._browse_image,
                  bg=BTN, fg=TXT, relief="flat", font=("Inter", 10), cursor="hand2",
                  padx=8).grid(row=1, column=2)

        # Buttons row
        btn_frame = tk.Frame(ctrl, bg=BG)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.run_btn = tk.Button(btn_frame, text="▶  Run Inference", command=self._run_async,
                                 bg=ACC, fg="white", relief="flat",
                                 font=("Inter", 12, "bold"), padx=20, pady=8, cursor="hand2")
        self.run_btn.pack(side="left", padx=8)

        tk.Button(btn_frame, text="💾  Save Result", command=self._save_result,
                  bg=BTN, fg=TXT, relief="flat",
                  font=("Inter", 11), padx=14, pady=8, cursor="hand2").pack(side="left", padx=8)

        # Stats bar
        self.stats_lbl = tk.Label(self, text="", bg=BG, fg="#aaa", font=("Inter", 10, "italic"))
        self.stats_lbl.pack()

        # ── Result canvas ─────────────────────────────────────────────────────
        self.canvas_frame = tk.Frame(self, bg=PANEL, bd=0)
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        self.canvas = tk.Canvas(self.canvas_frame, bg=PANEL, bd=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._canvas_img = None

        # Placeholder text
        self.canvas.create_text(
            550, 200,
            text="Upload an image and click ▶ Run Inference",
            fill="#555", font=("Inter", 14), tags="placeholder"
        )

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        path = self.model_path.get()
        if not os.path.exists(path):
            self.after(0, self._set_status, f"⚠  Model not found: {path}", "#f39c12")
            return
        try:
            m = ResUNet().to(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device))
            m.eval()
            self.model = m
            self.after(0, self._set_status,
                       f"✔  Model loaded on {self.device.type.upper()}", "#2ecc71")
        except Exception as e:
            self.after(0, self._set_status, f"✘  Load error: {e}", "#e74c3c")

    # ── Browsing helpers ──────────────────────────────────────────────────────
    def _browse_model(self):
        p = filedialog.askopenfilename(
            title="Select model weights",
            filetypes=[("PyTorch weights", "*.pth"), ("All files", "*.*")])
        if p:
            self.model_path.set(p)
            self._load_model_async()

    def _browse_image(self):
        p = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if p:
            self.image_path.set(p)

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_async(self):
        if not self.model:
            messagebox.showwarning("Model not loaded", "Please wait for the model to finish loading.")
            return
        if not self.image_path.get():
            messagebox.showwarning("No image", "Please select a test image first.")
            return
        self.run_btn.config(state="disabled", text="Running…")
        self._set_status("Running inference…", "#f39c12")
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        try:
            img_bgr = cv2.imread(self.image_path.get())
            if img_bgr is None:
                raise ValueError("Could not load image.")

            img_rgb, mask, overlay, skel_vis, seam_px = run_inference(
                img_bgr, self.model, self.device)

            result_pil = build_result_image(img_rgb, mask, overlay, skel_vis)
            self.result_img = result_pil

            self.after(0, self._show_result, result_pil, seam_px)
        except Exception as e:
            self.after(0, self._set_status, f"✘  Error: {e}", "#e74c3c")
            self.after(0, lambda: self.run_btn.config(state="normal", text="▶  Run Inference"))

    def _show_result(self, pil_img, seam_px):
        cw = self.canvas.winfo_width()  or 1060
        ch = self.canvas.winfo_height() or 380

        ratio     = min(cw / pil_img.width, ch / pil_img.height)
        new_w     = int(pil_img.width  * ratio)
        new_h     = int(pil_img.height * ratio)
        resized   = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self._canvas_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=self._canvas_img)

        self.stats_lbl.config(
            text=f"Seam pixels detected: {seam_px:,}  |  "
                 f"Coverage: {100 * seam_px / (pil_img.width * pil_img.height // 9):.2f}%  |  "
                 f"Device: {self.device.type.upper()}")

        self._set_status("✔  Inference complete", "#2ecc71")
        self.run_btn.config(state="normal", text="▶  Run Inference")

    # ── Save result ───────────────────────────────────────────────────────────
    def _save_result(self):
        if self.result_img is None:
            messagebox.showinfo("Nothing to save", "Run inference on an image first.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="weld_seam_result.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        if save_path:
            self.result_img.save(save_path)
            messagebox.showinfo("Saved", f"Result saved to:\n{save_path}")

    # ── Utility ───────────────────────────────────────────────────────────────
    def _set_status(self, msg, color="#eaeaea"):
        self.status_lbl.config(text=msg, fg=color)


if __name__ == "__main__":
    app = WeldSeamApp()
    app.mainloop()
