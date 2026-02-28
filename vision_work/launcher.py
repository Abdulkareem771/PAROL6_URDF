"""
Universal Launcher for WeldVision Test Tools
============================================
Provides a clean entry point to choose between the YOLO Object 
Detection tester and the PyTorch ResUNet Seam Segmentation tester.
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
    "accent": "#58a6ff",   # Blue
    "green":  "#3fb950",
    "warn":   "#d29922",
    "text":   "#c9d1d9",
    "text2":  "#8b949e",
}

class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WeldVision Launcher")
        self.geometry("400x320")
        self.resizable(False, False)
        self.configure(bg=C["bg"])
        
        # Center window
        self.eval('tk::PlaceWindow . center')
        
        self._build()

    def _build(self):
        # Header
        tk.Frame(self, bg=C["accent"], height=4).pack(fill="x")
        
        hdr = tk.Frame(self, bg=C["panel"])
        hdr.pack(fill="x", pady=(20, 30))
        tk.Label(hdr, text="WeldVision Toolkit", font=("Segoe UI", 18, "bold"), 
                 fg=C["text"], bg=C["panel"]).pack()
        tk.Label(hdr, text="Choose a testing module to launch", font=("Segoe UI", 10), 
                 fg=C["text2"], bg=C["panel"]).pack(pady=(4, 0))

        # Buttons
        self._btn("🔍  Detect Objects (YOLO)", self._launch_yolo, C["accent"]).pack(fill="x", padx=40, pady=10, ipady=8)
        self._btn("〰️  Segment Seam (ResUNet)", self._launch_resunet, C["green"]).pack(fill="x", padx=40, pady=10, ipady=8)

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

    def _launch_yolo(self):
        self._launch("yolo_training/yolo_gui.py")

    def _launch_resunet(self):
        self._launch("resunet_training/weld_seam_gui.py")

if __name__ == "__main__":
    Launcher().mainloop()
