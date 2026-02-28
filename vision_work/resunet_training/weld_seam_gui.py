"""
Weld Seam Detection — Premium GUI v3
3-Pane Layout:
  [Sidebar: Controls] | [File Browser: Folder Tree + Image Grid] | [Results: 3-Panel + Stats]

New features:
  • Full filesystem navigator with folder tree and clickable thumbnails
  • 3-panel Colab-style results (Original | View Mode | Skeleton always)
  • Live threshold slider with real-time mask refresh
  • Batch folder processing
  • Recent images history bar
  • Seam length estimation
  • Keyboard shortcut: Enter = Run, ← → = browse history, T = toggle view

Run: python3 weld_seam_gui.py
Deps: torch, opencv-python, scikit-image, Pillow, matplotlib, numpy
"""
import os, sys, threading, time, math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# ─────────────────────────── colour palette ──────────────────────────────────
C = dict(
    bg      = "#0d1117",
    panel   = "#161b22",
    card    = "#1c2128",
    border  = "#21262d",
    acc1    = "#58a6ff",   # blue
    acc2    = "#3fb950",   # green
    acc3    = "#f78166",   # coral
    warn    = "#d29922",   # amber
    txt     = "#c9d1d9",
    txt2    = "#8b949e",
    white   = "#ffffff",
    btn_bg  = "#21262d",
    sel     = "#1f4068",   # selected item bg
)
F = dict(h1=("Segoe UI",15,"bold"), h2=("Segoe UI",11,"bold"),
         body=("Segoe UI",10), sm=("Segoe UI",9), mono=("Courier New",9))

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".tiff",".tif",".webp"}

# ═══════════════════════════ ResUNet ═════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_c),nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1,bias=False),nn.BatchNorm2d(out_c))
        self.skip = nn.Sequential()
        if stride!=1 or in_c!=out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c,out_c,1,stride=stride,bias=False),nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x): return self.relu(self.conv(x)+self.skip(x))

class EncoderBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__(); self.res=ResBlock(in_c,out_c); self.pool=nn.MaxPool2d(2)
    def forward(self,x): s=self.res(x); return s,self.pool(s)

class Bridge(nn.Module):
    def __init__(self,in_c,out_c): super().__init__(); self.res=ResBlock(in_c,out_c)
    def forward(self,x): return self.res(x)

class DecoderBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_c,out_c,2,2); self.res=ResBlock(out_c*2,out_c)
    def forward(self,x,skip): return self.res(torch.cat([self.up(x),skip],1))

class ResUNet(nn.Module):
    def __init__(self,in_c=3,out_c=1):
        super().__init__()
        self.e1=EncoderBlock(in_c,64);   self.e2=EncoderBlock(64,128)
        self.e3=EncoderBlock(128,256);   self.e4=EncoderBlock(256,512)
        self.bridge=Bridge(512,1024)
        self.d4=DecoderBlock(1024,512);  self.d3=DecoderBlock(512,256)
        self.d2=DecoderBlock(256,128);   self.d1=DecoderBlock(128,64)
        self.head=nn.Conv2d(64,out_c,1)
    def forward(self,x):
        s1,x=self.e1(x); s2,x=self.e2(x); s3,x=self.e3(x); s4,x=self.e4(x)
        x=self.bridge(x); x=self.d4(x,s4); x=self.d3(x,s3)
        x=self.d2(x,s2); x=self.d1(x,s1); return self.head(x)

MEAN=np.array([0.485,0.456,0.406]); STD=np.array([0.229,0.224,0.225])

# ═══════════════════════════ Inference ═══════════════════════════════════════
def infer(img_bgr, model, device, thr=0.5):
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB); h,w=img_bgr.shape[:2]
    inp=cv2.resize(img_rgb,(512,512)); inp=(inp/255.0-MEAN)/STD
    t=torch.tensor(inp).float().permute(2,0,1).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad(): prob=torch.sigmoid(model(t)).squeeze().cpu().numpy()
    mask=(prob>thr).astype(np.uint8); mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
    prob_up=cv2.resize(prob,(w,h)); return img_rgb, mask, prob_up

def overlay_vis(img,mask,col=(220,60,60),a=0.55):
    out=img.copy().astype(np.float32); m=mask==1
    out[m]=(1-a)*out[m]+a*np.array(col); return np.clip(out,0,255).astype(np.uint8)

def skeleton_vis(img,mask):
    try:
        from skimage.morphology import skeletonize
        skel=skeletonize(mask); out=img.copy(); out[skel]=[0,220,60]; return out
    except ImportError:
        return overlay_vis(img,mask,(0,220,60))

def heatmap_vis(prob,img):
    norm=(prob*255).astype(np.uint8)
    heat=cv2.applyColorMap(norm,cv2.COLORMAP_INFERNO)
    heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img,0.4,heat,0.6,0)

def seam_length_px(mask):
    try:
        from skimage.morphology import skeletonize
        return int(skeletonize(mask).sum())
    except ImportError:
        return int(mask.sum())

def np2pil(arr): return Image.fromarray(arr.astype(np.uint8))

# ═══════════════════════════ Tiny shared widgets ══════════════════════════════
def _btn(p,text,cmd,bg=None,fg=None,font=None,pad=(10,5),**kw):
    bg=bg or C["btn_bg"]; fg=fg or C["txt"]; font=font or F["body"]
    b=tk.Button(p,text=text,command=cmd,bg=bg,fg=fg,
                activebackground=C["acc1"],activeforeground=C["white"],
                relief="flat",font=font,padx=pad[0],pady=pad[1],
                cursor="hand2",borderwidth=0,**kw)
    b.bind("<Enter>",lambda e:b.config(bg=C["border"]))
    b.bind("<Leave>",lambda e:b.config(bg=bg))
    return b

def _lbl(p,text,font=None,fg=None,bg=None,**kw):
    return tk.Label(p,text=text,font=font or F["body"],
                    fg=fg or C["txt"],bg=bg or C["bg"],**kw)

def _section_hdr(parent, text):
    f=tk.Frame(parent,bg=C["panel"]); f.pack(fill="x",padx=10,pady=(10,2))
    tk.Frame(f,bg=C["acc1"],width=3).pack(side="left",fill="y")
    _lbl(f,f"  {text}",font=F["sm"],fg=C["txt2"],bg=C["panel"]).pack(side="left")

# ═══════════════════════════ File Browser Panel ═══════════════════════════════
class FileBrowser(tk.Frame):
    """Left-of-results panel: folder tree + image thumbnails."""

    THUMB_SIZE = 90
    COLS       = 3

    def __init__(self, master, on_select, **kw):
        super().__init__(master, bg=C["panel"], **kw)
        self.on_select = on_select          # callback(path)
        self._thumb_refs = []               # keep PIL refs alive
        self._cur_dir    = os.path.expanduser("~")
        self._build()
        self._populate_tree(self._cur_dir)
        self._load_folder(self._cur_dir)

    # ── build ─────────────────────────────────────────────────────────────────
    def _build(self):
        # ── Header row ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["panel"])
        hdr.pack(fill="x", padx=6, pady=6)
        _lbl(hdr,"📁 File Browser",font=F["h2"],fg=C["white"],bg=C["panel"]).pack(side="left")
        _btn(hdr,"⬆ Up",self._go_up,pad=(6,3)).pack(side="right")
        _btn(hdr,"🏠",self._go_home,pad=(6,3)).pack(side="right",padx=4)

        # ── Path bar ────────────────────────────────────────────────────────
        pb = tk.Frame(self, bg=C["border"])
        pb.pack(fill="x", padx=6, pady=(0,4))
        self._path_var = tk.StringVar(value=self._cur_dir)
        pe = tk.Entry(pb, textvariable=self._path_var,
                      bg=C["border"], fg=C["acc1"], relief="flat",
                      font=F["mono"], insertbackground=C["txt"])
        pe.pack(fill="x", ipady=4, padx=4)
        pe.bind("<Return>", lambda e: self._navigate_to(self._path_var.get()))

        # ── Paned: tree (top) + grid (bottom) ───────────────────────────────
        paned = tk.PanedWindow(self, orient="vertical", bg=C["border"],
                               sashwidth=4, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=4)

        # Folder tree
        tree_frame = tk.Frame(paned, bg=C["panel"])
        paned.add(tree_frame, minsize=150, height=180)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("FB.Treeview", background=C["panel"], foreground=C["txt"],
                        fieldbackground=C["panel"], borderwidth=0,
                        font=F["sm"], rowheight=22)
        style.configure("FB.Treeview.Heading", background=C["border"], foreground=C["txt2"])
        style.map("FB.Treeview", background=[("selected", C["sel"])],
                  foreground=[("selected", C["white"])])

        self._tree = ttk.Treeview(tree_frame, style="FB.Treeview",
                                  show="tree", selectmode="browse")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<Double-1>", self._on_tree_double)

        # Image thumbnail grid
        grid_frame = tk.Frame(paned, bg=C["card"])
        paned.add(grid_frame, minsize=200)

        grid_hdr = tk.Frame(grid_frame, bg=C["card"])
        grid_hdr.pack(fill="x", padx=6, pady=4)
        _lbl(grid_hdr,"Images in folder",font=F["sm"],fg=C["txt2"],bg=C["card"]).pack(side="left")
        self._count_lbl = _lbl(grid_hdr,"",font=F["sm"],fg=C["txt2"],bg=C["card"])
        self._count_lbl.pack(side="right")

        canvas_wrap = tk.Frame(grid_frame, bg=C["card"])
        canvas_wrap.pack(fill="both", expand=True)
        self._grid_canvas = tk.Canvas(canvas_wrap, bg=C["card"], highlightthickness=0)
        gsb = ttk.Scrollbar(canvas_wrap, orient="vertical",
                             command=self._grid_canvas.yview)
        self._grid_canvas.configure(yscrollcommand=gsb.set)
        gsb.pack(side="right", fill="y")
        self._grid_canvas.pack(fill="both", expand=True)
        self._grid_inner = tk.Frame(self._grid_canvas, bg=C["card"])
        self._grid_canvas.create_window((0,0), window=self._grid_inner, anchor="nw")
        self._grid_inner.bind("<Configure>",
            lambda e: self._grid_canvas.configure(
                scrollregion=self._grid_canvas.bbox("all")))
        self._grid_canvas.bind("<MouseWheel>",
            lambda e: self._grid_canvas.yview_scroll(-1*(e.delta//120),"units"))

    # ── Folder tree ───────────────────────────────────────────────────────────
    def _populate_tree(self, path):
        self._tree.delete(*self._tree.get_children())
        # Root drives / home quick links
        roots = [("🏠  Home", os.path.expanduser("~")),
                 ("💻  Root", "/"),
                 ("🖥  Desktop", os.path.expanduser("~/Desktop"))]
        for label, p in roots:
            if os.path.exists(p):
                node = self._tree.insert("", "end", text=label, values=[p])
                self._tree.insert(node, "end", text="…")  # lazy placeholder

        self._insert_dir(None, path, expand=True)

    def _insert_dir(self, parent, path, expand=False):
        """Insert a directory node and lazily its children."""
        name = "📂  " + os.path.basename(path) if os.path.basename(path) else path
        node = self._tree.insert(parent or "", "end", text=name, values=[path])
        self._tree.insert(node, "end", text="…")
        if expand:
            self._tree.item(node, open=True)
            self._expand_node(node, path)
        return node

    def _expand_node(self, node, path):
        # Remove placeholder
        for child in self._tree.get_children(node):
            self._tree.delete(child)
        try:
            entries = sorted(
                [e for e in os.scandir(path) if e.is_dir() and not e.name.startswith(".")],
                key=lambda e: e.name.lower())
            for entry in entries[:40]:  # cap to 40 subdirs
                child = self._tree.insert(node, "end",
                                          text="📁  "+entry.name,
                                          values=[entry.path])
                self._tree.insert(child, "end", text="…")
        except PermissionError:
            pass

    def _on_tree_select(self, event):
        sel = self._tree.selection()
        if not sel: return
        vals = self._tree.item(sel[0], "values")
        if vals:
            path = vals[0]
            if os.path.isdir(path):
                self._expand_node(sel[0], path)
                self._load_folder(path)

    def _on_tree_double(self, event):
        self._on_tree_select(event)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _navigate_to(self, path):
        if os.path.isdir(path):
            self._cur_dir = path
            self._path_var.set(path)
            self._load_folder(path)
        elif os.path.isfile(path):
            self.on_select(path)

    def _go_up(self):
        parent = os.path.dirname(self._cur_dir)
        if parent != self._cur_dir:
            self._navigate_to(parent)

    def _go_home(self):
        self._navigate_to(os.path.expanduser("~"))

    # ── Thumbnail grid ────────────────────────────────────────────────────────
    def _load_folder(self, path):
        self._cur_dir = path
        self._path_var.set(path)
        # clear grid
        for w in self._grid_inner.winfo_children():
            w.destroy()
        self._thumb_refs.clear()

        try:
            imgs = sorted([f for f in os.listdir(path)
                           if os.path.splitext(f)[1].lower() in IMG_EXT])
        except PermissionError:
            return
        self._count_lbl.config(text=f"{len(imgs)} images")

        for idx, fname in enumerate(imgs):
            fpath = os.path.join(path, fname)
            col = idx % self.COLS
            row = idx // self.COLS
            cell = tk.Frame(self._grid_inner, bg=C["card"],
                            padx=3, pady=3)
            cell.grid(row=row, column=col, padx=3, pady=3)
            # Thumb canvas
            tc = tk.Canvas(cell, bg=C["border"], width=self.THUMB_SIZE,
                           height=self.THUMB_SIZE, highlightthickness=0,
                           cursor="hand2")
            tc.pack()
            name_lbl = _lbl(cell, fname[:14]+"…" if len(fname)>14 else fname,
                            font=F["sm"], fg=C["txt2"], bg=C["card"])
            name_lbl.pack()
            # Load thumb in background
            threading.Thread(target=self._load_thumb,
                             args=(tc, fpath), daemon=True).start()
            # Click to select
            tc.bind("<Button-1>", lambda e, p=fpath: self.on_select(p))
            tc.bind("<Enter>",
                lambda e, c=cell: c.config(bg=C["sel"]))
            tc.bind("<Leave>",
                lambda e, c=cell: c.config(bg=C["card"]))

    def _load_thumb(self, canvas, path):
        try:
            img = Image.open(path)
            img.thumbnail((self.THUMB_SIZE, self.THUMB_SIZE), Image.LANCZOS)
            ref = ImageTk.PhotoImage(img)
            self._thumb_refs.append(ref)
            self.after(0, lambda: self._place_thumb(canvas, ref))
        except Exception:
            pass

    def _place_thumb(self, canvas, ref):
        canvas.delete("all")
        canvas.create_image(self.THUMB_SIZE//2, self.THUMB_SIZE//2,
                            anchor="center", image=ref)


# ═══════════════════════════ Results Panel ═══════════════════════════════════
class ResultsPanel(tk.Frame):
    """Right area: 3-panel (Original / Mode View / Skeleton) + stats card."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=C["bg"], **kw)
        self._panels = {}
        self._build()

    def _build(self):
        # 3 equal image canvases
        top = tk.Frame(self, bg=C["bg"])
        top.pack(fill="both", expand=True)

        titles = ["Original", "Detection View", "Skeleton / Centerline"]
        for i, title in enumerate(titles):
            col_frame = tk.Frame(top, bg=C["card"])
            col_frame.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            hdr = tk.Frame(col_frame, bg=C["border"], height=28)
            hdr.pack(fill="x"); hdr.pack_propagate(False)
            _lbl(hdr, title, font=F["sm"], fg=C["txt2"], bg=C["border"]).pack(
                side="left", padx=8, pady=5)
            c = tk.Canvas(col_frame, bg=C["card"], highlightthickness=0)
            c.pack(fill="both", expand=True)
            c.create_text(10, 10, anchor="nw", text="—", fill=C["txt2"],
                          font=F["sm"], tags="placeholder")
            self._panels[title] = c
            c.bind("<Configure>", lambda e, k=title: self._on_resize(k))

        # Stats card at the bottom
        stats = tk.Frame(self, bg=C["panel"], height=90)
        stats.pack(fill="x", pady=(2, 0))
        stats.pack_propagate(False)
        self._build_stats_card(stats)

        # Store PIL refs
        self._pil_cache = {}
        self._img_refs  = {}

    def _build_stats_card(self, parent):
        inner = tk.Frame(parent, bg=C["panel"])
        inner.pack(fill="both", expand=True, padx=12, pady=8)

        self._stat_vars = {}
        fields = [
            ("Seam Pixels",  "seam_px",   C["acc1"]),
            ("Coverage",     "coverage",  C["acc2"]),
            ("Seam Length",  "length",    C["acc3"]),
            ("Inference",    "time",      C["warn"]),
            ("Image Size",   "size",      C["txt2"]),
            ("Threshold",    "thr",       C["txt2"]),
        ]
        for i, (label, key, color) in enumerate(fields):
            col = i % 3; row = i // 3
            cell = tk.Frame(inner, bg=C["panel"])
            cell.grid(row=row, column=col, padx=16, pady=2, sticky="w")
            _lbl(cell, label, font=F["sm"], fg=C["txt2"], bg=C["panel"]).pack(anchor="w")
            sv = tk.StringVar(value="—")
            self._stat_vars[key] = sv
            _lbl(cell, textvariable=sv, font=("Segoe UI", 11, "bold"),
                 fg=color, bg=C["panel"]).pack(anchor="w")

    def update_stats(self, **kw):
        for k, v in kw.items():
            if k in self._stat_vars:
                self._stat_vars[k].set(str(v))

    def show(self, key, pil_img):
        self._pil_cache[key] = pil_img
        self._render(key)

    def _render(self, key):
        if key not in self._pil_cache: return
        pil  = self._pil_cache[key]
        cv   = self._panels[key]
        cw   = cv.winfo_width()  or 300
        ch   = cv.winfo_height() or 300
        ratio = min(cw/pil.width, ch/pil.height)
        nw, nh = int(pil.width*ratio), int(pil.height*ratio)
        resized = pil.resize((nw, nh), Image.LANCZOS)
        ref = ImageTk.PhotoImage(resized)
        self._img_refs[key] = ref
        cv.delete("all")
        cv.create_image(cw//2, ch//2, anchor="center", image=ref)

    def _on_resize(self, key):
        self._render(key)

    def clear(self):
        for key, cv in self._panels.items():
            cv.delete("all")
            cv.create_text(10,10,anchor="nw",text="—",
                           fill=C["txt2"],font=F["sm"],tags="placeholder")
        self._pil_cache.clear()
        for key in self._stat_vars: self._stat_vars[key].set("—")


# ═══════════════════════════ Main Application ═════════════════════════════════
VIEW_MODES = ["Overlay", "Heat Map", "Mask Only"]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WeldVision  ·  ResUNet Seam Detector")
        self.geometry("1500x860")
        self.minsize(1100, 650)
        self.configure(bg=C["bg"])

        # State
        self.model      = None
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = tk.StringVar(value=self._default_model())
        self.threshold  = tk.DoubleVar(value=0.5)
        self.view_mode  = tk.StringVar(value="Overlay")

        self._img_path  = None
        self._orig_rgb  = None
        self._cur_mask  = None
        self._cur_prob  = None
        self._history   = []       # list of paths
        self._hist_idx  = -1
        self._scan      = False
        self._scan_ang  = 0.0

        self._build()
        self._bind_keys()
        self._load_model_async()

    def _default_model(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "best_resunet_seam.pth")

    # ══════════════ Build UI ══════════════════════════════════════════════════
    def _build(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        self._build_topbar()

        # ── Main 3-pane area ──────────────────────────────────────────────────
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)

        # Left: controls sidebar
        self._sidebar = tk.Frame(main, bg=C["panel"], width=230)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)
        self._build_sidebar()

        # Middle: file browser
        self._browser = FileBrowser(main, self._on_image_selected, width=320)
        self._browser.pack(side="left", fill="y")
        self._browser.pack_propagate(True)

        # Right: results
        self._results = ResultsPanel(main)
        self._results.pack(side="left", fill="both", expand=True)

        # ── Bottom status bar ─────────────────────────────────────────────────
        status = tk.Frame(self, bg=C["border"], height=28)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        self._status_lbl = _lbl(status, "Ready", font=F["sm"],
                                fg=C["txt2"], bg=C["border"])
        self._status_lbl.pack(side="left", padx=10, pady=6)
        _lbl(status,
             "Enter=Run  |  ←→=History  |  T=Toggle View  |  S=Save",
             font=F["sm"], fg=C["txt2"], bg=C["border"]).pack(side="right", padx=10)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=C["panel"], height=50)
        bar.pack(fill="x"); bar.pack_propagate(False)
        tk.Frame(bar, bg=C["acc1"], width=4).pack(side="left", fill="y")

        # Logo
        _lbl(bar, "  🔬 WeldVision", font=("Segoe UI",14,"bold"),
             fg=C["white"], bg=C["panel"]).pack(side="left", padx=8)
        _lbl(bar, "ResUNet Weld Seam Detection", font=F["sm"],
             fg=C["txt2"], bg=C["panel"]).pack(side="left", padx=4)

        # Right side info
        self._model_status_lbl = _lbl(bar, "⏳ Loading model…",
            font=F["sm"], fg=C["warn"], bg=C["panel"])
        self._model_status_lbl.pack(side="right", padx=16)

        dev_txt = f"Device: {self.device.type.upper()}"
        _lbl(bar, dev_txt, font=F["sm"], fg=C["acc2"], bg=C["panel"]).pack(
            side="right", padx=12)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        S = self._sidebar

        _section_hdr(S, "Model Weights")
        mf = tk.Frame(S, bg=C["panel"]); mf.pack(fill="x", padx=10, pady=4)
        tk.Entry(mf, textvariable=self.model_path, bg=C["border"], fg=C["txt"],
                 insertbackground=C["txt"], relief="flat",
                 font=F["mono"]).pack(fill="x", ipady=4)
        _btn(mf,"Browse .pth…",self._browse_model).pack(fill="x",pady=(4,0))

        _section_hdr(S, "Detection Threshold")
        tf = tk.Frame(S, bg=C["panel"]); tf.pack(fill="x", padx=10, pady=4)
        self._thresh_lbl = _lbl(tf, "0.50", font=("Segoe UI",12,"bold"),
                                 fg=C["acc1"], bg=C["panel"])
        self._thresh_lbl.pack(anchor="e")
        tk.Scale(tf, from_=0.01, to=0.99, resolution=0.01,
                 variable=self.threshold, orient="horizontal", length=210,
                 bg=C["panel"], fg=C["txt"], troughcolor=C["border"],
                 sliderrelief="flat", activebackground=C["acc1"],
                 highlightthickness=0, showvalue=False,
                 command=self._on_thr).pack(fill="x")
        _lbl(tf,"Low = sensitive  |  High = strict",
             font=F["sm"],fg=C["txt2"],bg=C["panel"]).pack()

        _section_hdr(S, "View Mode")
        vf = tk.Frame(S, bg=C["panel"]); vf.pack(fill="x", padx=10, pady=4)
        self._mode_btns = {}
        for m in VIEW_MODES:
            b=_btn(vf, m, lambda mv=m: self._set_view(mv))
            b.pack(fill="x", pady=2)
            self._mode_btns[m] = b
        self._set_view("Overlay")

        _section_hdr(S, "Actions")
        af = tk.Frame(S, bg=C["panel"]); af.pack(fill="x", padx=10, pady=4)

        self._run_btn = _btn(af, "▶  Run Inference", self._run_async,
                             bg=C["acc1"], fg=C["white"],
                             font=("Segoe UI",11,"bold"), pad=(10,8))
        self._run_btn.pack(fill="x", pady=2)
        self._run_btn.bind("<Enter>", lambda e: self._run_btn.config(bg="#79c0ff"))
        self._run_btn.bind("<Leave>", lambda e: self._run_btn.config(bg=C["acc1"]))

        _btn(af, "⏮ Prev Image", self._prev_hist).pack(fill="x", pady=2)
        _btn(af, "⏭ Next Image", self._next_hist).pack(fill="x", pady=2)
        _btn(af, "💾 Save Result", self._save).pack(fill="x", pady=2)
        _btn(af, "🗂 Batch Folder", self._batch).pack(fill="x", pady=2)

        _section_hdr(S, "Recent Images")
        self._hist_frame = tk.Frame(S, bg=C["panel"])
        self._hist_frame.pack(fill="x", padx=10, pady=4)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    def _bind_keys(self):
        self.bind("<Return>",   lambda e: self._run_async())
        self.bind("<Left>",     lambda e: self._prev_hist())
        self.bind("<Right>",    lambda e: self._next_hist())
        self.bind("t",          lambda e: self._cycle_view())
        self.bind("s",          lambda e: self._save())

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _on_image_selected(self, path):
        self._img_path = path
        self._set_status(f"Loaded: {os.path.basename(path)}")
        # Auto-run if model is ready
        if self.model:
            self._run_async()

    def _on_thr(self, val):
        self._thresh_lbl.config(text=f"{float(val):.2f}")
        if self._orig_rgb is not None:
            threading.Thread(target=self._refresh_view, daemon=True).start()

    def _browse_model(self):
        p = filedialog.askopenfilename(title="Select weights",
            filetypes=[("PyTorch weights","*.pth"),("All","*.*")])
        if p: self.model_path.set(p); self._load_model_async()

    def _set_view(self, mode):
        self.view_mode.set(mode)
        for m,b in self._mode_btns.items():
            b.config(bg=C["acc1"] if m==mode else C["btn_bg"],
                     fg=C["white"] if m==mode else C["txt"])
        if self._orig_rgb is not None:
            threading.Thread(target=self._refresh_view, daemon=True).start()

    def _cycle_view(self):
        modes = VIEW_MODES
        idx = modes.index(self.view_mode.get())
        self._set_view(modes[(idx+1)%len(modes)])

    # ── History navigation ────────────────────────────────────────────────────
    def _add_history(self, path):
        if path in self._history: self._history.remove(path)
        self._history.append(path)
        self._history = self._history[-10:]  # keep last 10
        self._hist_idx = len(self._history)-1
        self._refresh_hist_ui()

    def _refresh_hist_ui(self):
        for w in self._hist_frame.winfo_children(): w.destroy()
        for p in reversed(self._history[-5:]):
            name = os.path.basename(p)
            b = _btn(self._hist_frame,
                     ("▶ " if p==self._img_path else "  ") + (name[:18]+"…" if len(name)>18 else name),
                     lambda p=p: self._on_image_selected(p), pad=(4,3))
            b.pack(fill="x", pady=1)

    def _prev_hist(self):
        if self._hist_idx > 0:
            self._hist_idx -= 1
            self._on_image_selected(self._history[self._hist_idx])

    def _next_hist(self):
        if self._hist_idx < len(self._history)-1:
            self._hist_idx += 1
            self._on_image_selected(self._history[self._hist_idx])

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model_async(self):
        self._set_model_status("⏳ Loading…", C["warn"])
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        path = self.model_path.get()
        if not os.path.exists(path):
            self.after(0, self._set_model_status, "✘ Not found", C["acc3"]); return
        try:
            m = ResUNet().to(self.device)
            m.load_state_dict(torch.load(path, map_location=self.device)); m.eval()
            self.model = m
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            self.after(0, self._set_model_status,
                       f"✔ Ready · {params/1e6:.1f}M params", C["acc2"])
        except Exception as ex:
            self.after(0, self._set_model_status, f"✘ {ex}", C["acc3"])

    def _set_model_status(self, msg, col):
        self._model_status_lbl.config(text=msg, fg=col)

    # ── Inference ─────────────────────────────────────────────────────────────
    def _run_async(self):
        if not self.model:
            messagebox.showwarning("Model not ready","Please wait for the model to load."); return
        if not self._img_path or not os.path.exists(self._img_path):
            messagebox.showwarning("No image","Click an image in the file browser."); return
        self._run_btn.config(state="disabled", text="⏳  Analysing…")
        self._set_status(f"Running inference on {os.path.basename(self._img_path)}…")
        self._results.clear()
        self._start_spinner()
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        t0 = time.time()
        try:
            bgr = cv2.imread(self._img_path)
            if bgr is None: raise ValueError("Cannot read image")
            thr = self.threshold.get()
            rgb, mask, prob = infer(bgr, self.model, self.device, thr)
            self._orig_rgb = rgb; self._cur_mask = mask; self._cur_prob = prob
            elapsed = time.time()-t0
            self.after(0, self._show_results, elapsed)
        except Exception as ex:
            self.after(0, self._inference_err, str(ex))

    def _show_results(self, elapsed):
        self._stop_spinner()
        mask = self._cur_mask; rgb = self._orig_rgb; prob = self._cur_prob
        h,w = mask.shape; px = int(mask.sum())
        length = seam_length_px(mask)

        # Always show original
        self._results.show("Original", np2pil(rgb))
        # Middle: view mode
        mid = self._build_mid(rgb, mask, prob)
        self._results.show("Detection View", np2pil(mid))
        # Right: always skeleton
        self._results.show("Skeleton / Centerline", np2pil(skeleton_vis(rgb, mask)))

        self._results.update_stats(
            seam_px=f"{px:,}",
            coverage=f"{100*px/(h*w):.3f} %",
            length=f"~{length:,} px",
            time=f"{elapsed*1000:.0f} ms",
            size=f"{w}×{h}",
            thr=f"{self.threshold.get():.2f}",
        )
        self._run_btn.config(state="normal", text="▶  Run Inference")
        self._set_status(f"✔  Done — {px:,} seam pixels detected in {elapsed*1000:.0f} ms")
        self._add_history(self._img_path)

    def _build_mid(self, rgb, mask, prob):
        mode = self.view_mode.get()
        if   mode == "Overlay":   return overlay_vis(rgb, mask)
        elif mode == "Heat Map":  return heatmap_vis(prob, rgb)
        elif mode == "Mask Only": return (mask*255)[...,None].repeat(3,axis=2)
        return overlay_vis(rgb, mask)

    def _refresh_view(self):
        if self._cur_prob is None: return
        thr = self.threshold.get()
        mask = (self._cur_prob > thr).astype(np.uint8)
        self._cur_mask = mask
        rgb = self._orig_rgb; prob = self._cur_prob; h,w=mask.shape
        px = int(mask.sum()); length = seam_length_px(mask)
        mid = self._build_mid(rgb, mask, prob)
        skel = skeleton_vis(rgb, mask)
        self.after(0, self._results.show, "Detection View", np2pil(mid))
        self.after(0, self._results.show, "Skeleton / Centerline", np2pil(skel))
        self.after(0, self._results.update_stats,
                   **dict(seam_px=f"{px:,}",coverage=f"{100*px/(h*w):.3f} %",
                          length=f"~{length:,} px",thr=f"{thr:.2f}"))

    def _inference_err(self, msg):
        self._stop_spinner()
        messagebox.showerror("Error", msg)
        self._run_btn.config(state="normal", text="▶  Run Inference")

    # ── Spinner (top-bar pulse) ────────────────────────────────────────────────
    def _start_spinner(self):
        self._scan=True; self._pulse()
    def _stop_spinner(self):
        self._scan=False; self._model_status_lbl.config(text=
            self._model_status_lbl.cget("text").replace("⏳","✔"))

    def _pulse(self):
        if not self._scan: return
        dots="⣾⣽⣻⢿⡿⣟⣯⣷"
        self._scan_ang=(self._scan_ang+1)%len(dots)
        self._set_status(f"{dots[int(self._scan_ang)]}  Analysing…")
        self.after(80, self._pulse)

    # ── Batch processing ──────────────────────────────────────────────────────
    def _batch(self):
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder: return
        out_dir = os.path.join(folder, "seam_results")
        os.makedirs(out_dir, exist_ok=True)
        imgs = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXT]
        if not imgs: messagebox.showinfo("No images","No images in that folder."); return
        self._set_status(f"Batch: 0/{len(imgs)}")
        self._run_btn.config(state="disabled", text="⏳ Batch…")
        threading.Thread(target=self._run_batch, args=(folder,imgs,out_dir), daemon=True).start()

    def _run_batch(self, folder, imgs, out_dir):
        thr = self.threshold.get()
        for i, fname in enumerate(imgs):
            path = os.path.join(folder, fname)
            try:
                bgr = cv2.imread(path)
                if bgr is None: continue
                rgb, mask, prob = infer(bgr, self.model, self.device, thr)
                ov  = overlay_vis(rgb, mask)
                sk  = skeleton_vis(rgb, mask)
                combined = np.concatenate([rgb, ov, sk], axis=1)
                out_path = os.path.join(out_dir, os.path.splitext(fname)[0]+"_result.png")
                cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                self.after(0, self._set_status, f"Batch: {i+1}/{len(imgs)} — {fname}")
            except Exception: pass
        self.after(0, self._run_btn.config, {"state":"normal","text":"▶  Run Inference"})
        self.after(0, messagebox.showinfo, "Batch complete",
                   f"Saved {len(imgs)} results to:\n{out_dir}")

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self):
        if not self._cur_mask is not None and self._orig_rgb is not None:
            messagebox.showinfo("Nothing to save","Run inference first."); return
        rgb=self._orig_rgb; mask=self._cur_mask; prob=self._cur_prob
        if rgb is None: messagebox.showinfo("Nothing to save","Run inference first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile="seam_result.png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")])
        if not path: return
        ov = overlay_vis(rgb, mask)
        sk = skeleton_vis(rgb, mask)
        combined = np.concatenate([rgb, ov, sk], axis=1)
        cv2.imwrite(path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Saved", f"3-panel result saved to:\n{path}")

    # ── Status ────────────────────────────────────────────────────────────────
    def _set_status(self, msg):
        self._status_lbl.config(text=f"  {msg}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
