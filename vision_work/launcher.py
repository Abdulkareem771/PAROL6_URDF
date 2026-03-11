"""
Universal Launcher for WeldVision Test Tools
============================================
Scrollable launcher — all tools fit no matter how many are added.
"""
import os
import sys
import tkinter as tk
import subprocess
from tkinter import messagebox

# ─── Colors ───────────────────────────────────────────────────────────────────
C = {
    "bg":     "#111318",
    "panel":  "#1e2229",
    "border": "#2d333b",
    "accent": "#58a6ff",
    "green":  "#3fb950",
    "warn":   "#d29922",
    "text":   "#c9d1d9",
    "text2":  "#8b949e",
}


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WeldVision Launcher")
        self.geometry("420x700")
        self.minsize(420, 400)
        self.configure(bg=C["bg"])
        self.eval('tk::PlaceWindow . center')
        self._build()

    def _build(self):
        # ── Fixed Header ──────────────────────────────────────────────────────
        tk.Frame(self, bg=C["accent"], height=4).pack(fill="x")
        hdr = tk.Frame(self, bg=C["panel"])
        hdr.pack(fill="x", pady=(16, 10))
        tk.Label(hdr, text="WeldVision Toolkit", font=("Segoe UI", 18, "bold"),
                 fg=C["text"], bg=C["panel"]).pack()
        tk.Label(hdr, text="Choose a testing module to launch", font=("Segoe UI", 10),
                 fg=C["text2"], bg=C["panel"]).pack(pady=(4, 0))

        # ── Scrollable Body ───────────────────────────────────────────────────
        container = tk.Frame(self, bg=C["bg"])
        container.pack(fill="both", expand=True, padx=0, pady=(0, 0))

        canvas = tk.Canvas(container, bg=C["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview,
                                 bg=C["panel"], troughcolor=C["border"])
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Inner frame that holds all buttons
        inner = tk.Frame(canvas, bg=C["bg"])
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ── Legacy Tools ──────────────────────────────────────────────────────
        self._section(inner, "LEGACY TOOLS")
        self._btn_in(inner, "🔍  Detect Objects (YOLO)", self._launch_yolo, C["accent"])
        self._btn_in(inner, "〰️  Segment Seam (ResUNet)", self._launch_resunet, C["green"])
        self._btn_in(inner, "🖍️  Manual Path Annotator", self._launch_annotator, "#a371f7")

        # ── Next-Gen Tools ────────────────────────────────────────────────────
        self._section(inner, "NEXT-GEN TOOLS")
        self._btn_in(inner, "🔮  Pipeline Prototyper", self._launch_prototyper, "#e3b341")
        self._btn_in(inner, "🐠  Script Sandbox", self._launch_script_sandbox, "#f38ba8")
        self._btn_in(inner, "🎨  Mask Painter", self._launch_mask_painter, "#cba6f7")
        self._btn_in(inner, "🔍  YOLO Inspector", self._launch_yolo_inspector, "#89dceb")
        self._btn_in(inner, "〰️  Seam Inspector", self._launch_seam_inspector, "#a6e3a1")
        self._btn_in(inner, "🧪  Mask Pipeline Tester", self._launch_mask_pipeline, "#fab387")
        self._btn_in(inner, "🎬  Pipeline Studio", self._launch_pipeline_studio, "#94e2d5")
        self._btn_in(inner, "✏️  Annotation Studio", self._launch_annotation_studio, "#eba0ac")
        self._btn_in(inner, "📦  Batch YOLO Exporter", self._launch_batch_yolo, "#b4befe")

        # Bottom padding
        tk.Frame(inner, bg=C["bg"], height=20).pack()

    def _section(self, parent, label):
        """Visual section separator with label."""
        frm = tk.Frame(parent, bg=C["bg"])
        frm.pack(fill="x", padx=20, pady=(12, 4))
        tk.Frame(frm, bg=C["border"], height=1).pack(fill="x")
        tk.Label(frm, text=label, font=("Segoe UI", 8, "bold"),
                 fg=C["text2"], bg=C["bg"]).pack(anchor="w", padx=4, pady=(2, 0))

    def _btn_in(self, parent, text, cmd, bg_color):
        """Button packed inside the scrollable inner frame."""
        tk.Button(parent, text=text, command=cmd, bg=bg_color, fg="#ffffff",
                  font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
                  borderwidth=0, padx=8, pady=8).pack(
            fill="x", padx=40, pady=3)

    # ── kept for backward compat if referenced somewhere ─────────────────────
    def _btn(self, text, cmd, bg_color):
        return tk.Button(self, text=text, command=cmd, bg=bg_color, fg="#ffffff",
                         font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2", borderwidth=0)

    def _launch(self, script_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, script_path)
        if not os.path.exists(full_path):
            messagebox.showerror("Error", f"Could not find script:\n{full_path}")
            return
        print(f"Launching {full_path}...")
        subprocess.Popen([sys.executable, full_path])
        self.destroy()

    # ── Launch methods ────────────────────────────────────────────────────────
    def _launch_yolo(self):               self._launch("yolo_training/yolo_gui.py")
    def _launch_resunet(self):            self._launch("resunet_training/weld_seam_gui.py")
    def _launch_annotator(self):          self._launch("tools/manual_annotator.py")
    def _launch_prototyper(self):         self._launch("tools/pipeline_prototyper.py")
    def _launch_script_sandbox(self):     self._launch("tools/script_sandbox.py")
    def _launch_mask_painter(self):       self._launch("tools/mask_painter.py")
    def _launch_yolo_inspector(self):     self._launch("tools/yolo_inspector.py")
    def _launch_seam_inspector(self):     self._launch("tools/seam_inspector.py")
    def _launch_mask_pipeline(self):      self._launch("tools/mask_pipeline_tester.py")
    def _launch_pipeline_studio(self):    self._launch("tools/pipeline_studio.py")
    def _launch_annotation_studio(self):  self._launch("tools/annotation_studio.py")
    def _launch_batch_yolo(self):         self._launch("tools/batch_yolo.py")


if __name__ == "__main__":
    Launcher().mainloop()
