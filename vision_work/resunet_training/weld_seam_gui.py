"""
WeldVision GUI v4
=================
Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Top bar: [Open Image] [Run] [Save]   Model status   Device     │
  ├──────────┬──────────────────────────────────────────────────────┤
  │  Sidebar │  [Original]  |  [Detection]  |  [Skeleton]           │
  │  Controls│                                                       │
  │          │                                                       │
  │          │                                                       │
  ├──────────┴──────────────────────────────────────────────────────┤
  │  View: [Overlay] [Mask] [Skeleton] [Heat Map]    Stats cards    │
  └─────────────────────────────────────────────────────────────────┘

Image picking opens a SEPARATE popup window (Toplevel) with a full
filesystem file browser — click an image, window closes, path is set.
The file browser starts at /host_home (mounted host /home/kareem) to
give access to the full computer, and /workspace for the project.

Run: python3 weld_seam_gui.py
"""
import os, sys, threading, time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2, numpy as np

try:
    import torch, torch.nn as nn
except ImportError:
    print("pip install torch --index-url https://download.pytorch.org/whl/cpu"); sys.exit(1)

# ─── Colours & Fonts ──────────────────────────────────────────────────────────
BG     = "#f0f2f5"
WHITE  = "#ffffff"
DARK   = "#1a1f2e"
PANEL  = "#ffffff"
BORDER = "#e1e4e8"
ACC    = "#2563eb"       # blue
GREEN  = "#16a34a"
RED    = "#dc2626"
AMBER  = "#d97706"
GREY   = "#6b7280"
SEL    = "#dbeafe"

FH1  = ("Segoe UI", 14, "bold")
FH2  = ("Segoe UI", 11, "bold")
FBODY= ("Segoe UI", 10)
FSM  = ("Segoe UI", 9)
FMONO= ("Courier New", 9)

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

# ─── ResUNet ─────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(i,o,3,s,1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),
                               nn.Conv2d(o,o,3,1,1,bias=False),nn.BatchNorm2d(o))
        self.sk= nn.Sequential() if (s==1 and i==o) else nn.Sequential(
                    nn.Conv2d(i,o,1,s,bias=False),nn.BatchNorm2d(o))
        self.r = nn.ReLU(True)
    def forward(self,x): return self.r(self.c(x)+self.sk(x))

class Enc(nn.Module):
    def __init__(self,i,o): super().__init__(); self.r=ResBlock(i,o); self.p=nn.MaxPool2d(2)
    def forward(self,x): s=self.r(x); return s,self.p(s)

class Dec(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.u=nn.ConvTranspose2d(i,o,2,2); self.r=ResBlock(o*2,o)
    def forward(self,x,s): return self.r(torch.cat([self.u(x),s],1))

class ResUNet(nn.Module):
    def __init__(self,i=3,o=1):
        super().__init__()
        self.e1=Enc(i,64);self.e2=Enc(64,128);self.e3=Enc(128,256);self.e4=Enc(256,512)
        self.b=ResBlock(512,1024)
        self.d4=Dec(1024,512);self.d3=Dec(512,256);self.d2=Dec(256,128);self.d1=Dec(128,64)
        self.h=nn.Conv2d(64,o,1)
    def forward(self,x):
        s1,x=self.e1(x);s2,x=self.e2(x);s3,x=self.e3(x);s4,x=self.e4(x)
        x=self.b(x);x=self.d4(x,s4);x=self.d3(x,s3);x=self.d2(x,s2);x=self.d1(x,s1)
        return self.h(x)

M = np.array([0.485,0.456,0.406]); S = np.array([0.229,0.224,0.225])

def run_model(bgr, model, device, thr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); h,w=bgr.shape[:2]
    t=(cv2.resize(rgb,(512,512))/255.0-M)/S
    t=torch.tensor(t).float().permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad(): p=torch.sigmoid(model(t)).squeeze().cpu().numpy()
    mask=(p>thr).astype(np.uint8); return rgb,cv2.resize(mask,(w,h),cv2.INTER_NEAREST),cv2.resize(p,(w,h))

def skel_len(mask):
    try:
        from skimage.morphology import skeletonize; return int(skeletonize(mask).sum())
    except: return int(mask.sum())

def make_overlay(rgb,mask,c=(220,55,55),a=0.55):
    o=rgb.copy().astype(float); o[mask==1]=(1-a)*o[mask==1]+a*np.array(c)
    return np.clip(o,0,255).astype(np.uint8)

def make_skel(rgb,mask):
    try:
        from skimage.morphology import skeletonize
        sk=skeletonize(mask); o=rgb.copy(); o[sk]=[0,210,60]; return o
    except: return make_overlay(rgb,mask,(0,210,60))

def make_heat(prob,rgb):
    h=cv2.applyColorMap((prob*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(rgb,0.35,cv2.cvtColor(h,cv2.COLOR_BGR2RGB),0.65,0)

def make_mask_only(mask):
    return (np.stack([mask]*3,axis=2)*255).astype(np.uint8)

def np2pil(a): return Image.fromarray(a.astype(np.uint8))

# ─── Popup Image Picker ───────────────────────────────────────────────────────
class ImagePicker(tk.Toplevel):
    """Separate window: folder tree + thumbnail grid. Returns path on selection."""

    THUMB = 96
    COLS  = 4

    # Root paths visible inside Docker — ordered by priority
    ROOTS = [
        ("💻 Host Home",  "/host_home"),
        ("📁 Workspace",  "/workspace"),
        ("🖥 Root",        "/"),
    ]

    def __init__(self, master, callback):
        super().__init__(master)
        self.title("Open Image")
        self.geometry("880x620")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.callback  = callback
        self._thumbs   = []
        self._cur_dir  = self._best_root()
        self._build()
        self._fill_tree()
        self._load_grid(self._cur_dir)
        self.grab_set()          # modal

    def _best_root(self):
        for _, p in self.ROOTS:
            if os.path.isdir(p): return p
        return "/"

    def _build(self):
        # Top bar
        top = tk.Frame(self, bg=DARK, height=44); top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top,text="  📂  Select an Image",font=FH2,bg=DARK,fg=WHITE).pack(side="left",pady=10)
        tk.Button(top,text="✕ Cancel",command=self.destroy,bg=DARK,fg=GREY,
                  relief="flat",font=FSM,cursor="hand2").pack(side="right",padx=10)

        # Path bar
        pb = tk.Frame(self, bg=BORDER, height=32); pb.pack(fill="x"); pb.pack_propagate(False)
        self._path_sv = tk.StringVar(value=self._cur_dir)
        pe = tk.Entry(pb,textvariable=self._path_sv,bg=WHITE,fg=DARK,relief="flat",
                      font=FMONO,insertbackground=DARK)
        pe.pack(fill="x",ipady=6,padx=4,pady=2)
        pe.bind("<Return>",lambda e: self._nav(self._path_sv.get()))
        tk.Button(pb,text="⬆",command=self._up,bg=BORDER,fg=DARK,
                  relief="flat",cursor="hand2",font=FBODY).place(relx=1,rely=0,anchor="ne")

        # Quick‑access buttons
        qf = tk.Frame(self, bg=BG); qf.pack(fill="x", padx=8, pady=4)
        for label, path in self.ROOTS:
            if os.path.isdir(path):
                b=tk.Button(qf,text=label,command=lambda p=path:self._nav(p),
                            bg=WHITE,fg=DARK,relief="flat",font=FSM,
                            padx=10,pady=4,cursor="hand2",bd=1)
                b.pack(side="left",padx=4)

        # Paned: tree | grid
        paned = tk.PanedWindow(self,orient="horizontal",bg=BORDER,sashwidth=4)
        paned.pack(fill="both",expand=True,padx=8,pady=4)

        # Folder tree
        tf = tk.Frame(paned,bg=WHITE); paned.add(tf,minsize=220,width=240)
        style=ttk.Style(); style.theme_use("clam")
        style.configure("P.Treeview",background=WHITE,foreground=DARK,
                        fieldbackground=WHITE,font=FSM,rowheight=24,borderwidth=0)
        style.map("P.Treeview",background=[("selected",SEL)],foreground=[("selected",DARK)])
        self._tree=ttk.Treeview(tf,style="P.Treeview",show="tree",selectmode="browse")
        sb=ttk.Scrollbar(tf,orient="vertical",command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right",fill="y"); self._tree.pack(fill="both",expand=True)
        self._tree.bind("<<TreeviewSelect>>",self._on_tree)

        # Thumbnail grid
        gf = tk.Frame(paned,bg=WHITE); paned.add(gf,minsize=400)
        hdr = tk.Frame(gf,bg=BG); hdr.pack(fill="x",padx=6,pady=4)
        tk.Label(hdr,text="Images",font=FH2,bg=BG,fg=DARK).pack(side="left")
        self._cnt_lbl=tk.Label(hdr,text="",font=FSM,bg=BG,fg=GREY); self._cnt_lbl.pack(side="right")
        wrap=tk.Frame(gf,bg=WHITE); wrap.pack(fill="both",expand=True)
        self._gc=tk.Canvas(wrap,bg=WHITE,highlightthickness=0)
        gsb=ttk.Scrollbar(wrap,orient="vertical",command=self._gc.yview)
        self._gc.configure(yscrollcommand=gsb.set)
        gsb.pack(side="right",fill="y"); self._gc.pack(fill="both",expand=True)
        self._gi=tk.Frame(self._gc,bg=WHITE)
        self._gc.create_window((0,0),window=self._gi,anchor="nw")
        self._gi.bind("<Configure>",lambda e:self._gc.configure(
            scrollregion=self._gc.bbox("all")))
        self._gc.bind("<MouseWheel>",lambda e:self._gc.yview_scroll(-e.delta//120,"units"))

    def _fill_tree(self):
        self._tree.delete(*self._tree.get_children())
        for label, path in self.ROOTS:
            if os.path.isdir(path):
                n=self._tree.insert("","end",text=label,values=[path])
                self._tree.insert(n,"end",text="…")
        # also insert current dir expanded
        n=self._tree.insert("","end",text="📂 "+os.path.basename(self._cur_dir),
                             values=[self._cur_dir])
        self._expand(n,self._cur_dir)
        self._tree.item(n,open=True)

    def _expand(self, node, path):
        for c in self._tree.get_children(node): self._tree.delete(c)
        try:
            dirs=sorted([e for e in os.scandir(path) if e.is_dir() and not e.name.startswith(".")],
                         key=lambda e:e.name.lower())
            for d in dirs[:50]:
                c=self._tree.insert(node,"end",text="📁 "+d.name,values=[d.path])
                self._tree.insert(c,"end",text="…")
        except PermissionError: pass

    def _on_tree(self, _):
        sel=self._tree.selection()
        if not sel: return
        vals=self._tree.item(sel[0],"values")
        if vals:
            p=vals[0]
            if os.path.isdir(p):
                self._expand(sel[0],p)
                self._load_grid(p)

    def _nav(self, path):
        if os.path.isdir(path): self._cur_dir=path; self._path_sv.set(path); self._load_grid(path)
        elif os.path.isfile(path): self._select(path)

    def _up(self):
        p=os.path.dirname(self._cur_dir)
        if p!=self._cur_dir: self._nav(p)

    def _load_grid(self, path):
        self._cur_dir=path; self._path_sv.set(path)
        for w in self._gi.winfo_children(): w.destroy()
        self._thumbs.clear()
        try:
            imgs=sorted([f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in IMG_EXT])
        except PermissionError: return
        self._cnt_lbl.config(text=f"{len(imgs)} images")
        for i,fname in enumerate(imgs):
            fp=os.path.join(path,fname)
            col=i%self.COLS; row=i//self.COLS
            cell=tk.Frame(self._gi,bg=WHITE,padx=4,pady=4)
            cell.grid(row=row,column=col,padx=4,pady=4)
            c=tk.Canvas(cell,bg=BORDER,width=self.THUMB,height=self.THUMB,
                        highlightthickness=2,highlightbackground=BORDER,cursor="hand2")
            c.pack()
            tk.Label(cell,text=fname[:13]+("|" if len(fname)>13 else ""),
                     font=FSM,bg=WHITE,fg=GREY,wraplength=self.THUMB).pack()
            threading.Thread(target=self._load_thumb,args=(c,fp),daemon=True).start()
            c.bind("<Button-1>",lambda e,p=fp:self._select(p))
            c.bind("<Enter>",lambda e,c=c:c.config(highlightbackground=ACC))
            c.bind("<Leave>",lambda e,c=c:c.config(highlightbackground=BORDER))

    def _load_thumb(self,canvas,path):
        try:
            img=Image.open(path); img.thumbnail((self.THUMB,self.THUMB),Image.LANCZOS)
            ref=ImageTk.PhotoImage(img); self._thumbs.append(ref)
            self.after(0,lambda c=canvas,r=ref:self._place(c,r))
        except: pass

    def _place(self,canvas,ref):
        canvas.delete("all")
        canvas.create_image(self.THUMB//2,self.THUMB//2,anchor="center",image=ref)

    def _select(self,path):
        self.callback(path); self.destroy()


# ─── Main Application ─────────────────────────────────────────────────────────
VIEWS = ["Overlay","Mask","Heat Map","Skeleton"]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WeldVision · ResUNet Seam Detector")
        self.geometry("1380x820")
        self.minsize(1000,640)
        self.configure(bg=BG)

        self.model    = None
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mpath   = tk.StringVar(value=self._def_model())
        self._thr     = tk.DoubleVar(value=0.50)
        self._view    = tk.StringVar(value="Overlay")

        self._img_path = None
        self._rgb      = None
        self._mask     = None
        self._prob     = None
        self._hist     = []

        self._build()
        self._bind_keys()
        threading.Thread(target=self._load_model,daemon=True).start()

    def _def_model(self):
        here=os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here,"best_resunet_seam.pth")

    # ── build ──────────────────────────────────────────────────────────────────
    def _build(self):
        self._build_topbar()
        body=tk.Frame(self,bg=BG); body.pack(fill="both",expand=True,padx=10,pady=(4,0))
        self._build_sidebar(body)
        self._build_viewer(body)
        self._build_controlbar()

    # ── Top bar ────────────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar=tk.Frame(self,bg=DARK,height=52); bar.pack(fill="x"); bar.pack_propagate(False)
        tk.Frame(bar,bg=ACC,width=5).pack(side="left",fill="y")
        tk.Label(bar,text="  🔬 WeldVision",font=FH1,bg=DARK,fg=WHITE).pack(side="left",padx=4)
        tk.Label(bar,text="ResUNet Weld Seam Detection",font=FSM,bg=DARK,fg=GREY).pack(side="left",padx=6,pady=14)

        # Right cluster
        r=tk.Frame(bar,bg=DARK); r.pack(side="right",padx=12)
        self._dev_lbl=tk.Label(r,text=f"⚡ {self.device.type.upper()}",
                               font=FSM,bg=DARK,fg=GREEN); self._dev_lbl.pack(anchor="e")
        self._mdl_lbl=tk.Label(r,text="⏳ Loading model…",
                               font=FSM,bg=DARK,fg=AMBER); self._mdl_lbl.pack(anchor="e")

        # Action buttons in top bar
        for text,cmd,bg in [
            ("📂  Open Image", self._open_picker, ACC),
            ("▶   Run",        self._run,         "#16a34a"),
            ("💾  Save",        self._save,        "#374151"),
        ]:
            b=tk.Button(bar,text=text,command=cmd,bg=bg,fg=WHITE,
                        relief="flat",font=("Segoe UI",10,"bold"),
                        padx=14,pady=6,cursor="hand2",borderwidth=0)
            b.pack(side="left",padx=4,pady=8)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    def _build_sidebar(self,parent):
        sb=tk.Frame(parent,bg=WHITE,width=200,bd=1,relief="flat")
        sb.pack(side="left",fill="y",padx=(0,8))
        sb.pack_propagate(False)
        tk.Frame(sb,bg=BORDER,height=1).pack(fill="x")

        # Model
        self._sec(sb,"Model")
        mf=tk.Frame(sb,bg=WHITE); mf.pack(fill="x",padx=10,pady=4)
        tk.Entry(mf,textvariable=self._mpath,bg=BG,fg=DARK,relief="flat",
                 font=FMONO).pack(fill="x",ipady=3)
        self._mkbtn(mf,"Browse .pth…",self._browse_model).pack(fill="x",pady=(4,0))

        # Threshold
        self._sec(sb,"Threshold")
        tf=tk.Frame(sb,bg=WHITE); tf.pack(fill="x",padx=10,pady=4)
        self._thr_lbl=tk.Label(tf,text="0.50",font=("Segoe UI",13,"bold"),
                                bg=WHITE,fg=ACC); self._thr_lbl.pack(anchor="e")
        tk.Scale(tf,from_=0.01,to=0.99,resolution=0.01,variable=self._thr,
                 orient="horizontal",length=180,bg=WHITE,fg=DARK,
                 troughcolor=BORDER,sliderrelief="flat",activebackground=ACC,
                 highlightthickness=0,showvalue=False,
                 command=lambda v:(self._thr_lbl.config(text=f"{float(v):.2f}"),
                                   self._refresh() if self._rgb is not None else None)
                 ).pack(fill="x")
        tk.Label(tf,text="Low=sensitive  High=strict",font=FSM,bg=WHITE,fg=GREY).pack()

        # History
        self._sec(sb,"Recent Images")
        self._hist_frame=tk.Frame(sb,bg=WHITE); self._hist_frame.pack(fill="x",padx=10,pady=4)

        # Stats
        self._sec(sb,"Detection Stats")
        sf=tk.Frame(sb,bg=WHITE); sf.pack(fill="x",padx=10,pady=4)
        self._svars={}
        for key,label in [("px","Seam pixels"),("cov","Coverage"),
                           ("len","Seam length"),("ms","Inference")]:
            row=tk.Frame(sf,bg=WHITE); row.pack(fill="x",pady=2)
            tk.Label(row,text=label,font=FSM,bg=WHITE,fg=GREY).pack(anchor="w")
            sv=tk.StringVar(value="—"); self._svars[key]=sv
            tk.Label(row,textvariable=sv,font=("Segoe UI",11,"bold"),
                     bg=WHITE,fg=DARK).pack(anchor="w")

    def _sec(self,parent,text):
        f=tk.Frame(parent,bg=WHITE); f.pack(fill="x",padx=10,pady=(10,2))
        tk.Frame(f,bg=ACC,width=3).pack(side="left",fill="y")
        tk.Label(f,text=f"  {text}",font=("Segoe UI",9,"bold"),
                 bg=WHITE,fg=GREY).pack(side="left")

    def _mkbtn(self,p,t,cmd,bg=BG,fg=DARK):
        return tk.Button(p,text=t,command=cmd,bg=bg,fg=fg,relief="flat",
                         font=FSM,padx=8,pady=4,cursor="hand2",borderwidth=0)

    # ── Viewer: 3 panels ───────────────────────────────────────────────────────
    def _build_viewer(self,parent):
        self._viewer=tk.Frame(parent,bg=BG); self._viewer.pack(fill="both",expand=True)
        self._canvases={}
        self._pil_cache={}
        self._img_refs={}
        for title in ("Original","Detection View","Skeleton / Centerline"):
            col=tk.Frame(self._viewer,bg=WHITE,bd=1,relief="flat")
            col.pack(side="left",fill="both",expand=True,padx=4,pady=2)
            hdr=tk.Frame(col,bg=BG,height=28); hdr.pack(fill="x"); hdr.pack_propagate(False)
            tk.Label(hdr,text=title,font=("Segoe UI",10,"bold"),
                     bg=BG,fg=DARK).pack(side="left",padx=10,pady=6)
            c=tk.Canvas(col,bg=WHITE,highlightthickness=0)
            c.pack(fill="both",expand=True)
            c.create_text(10,20,anchor="nw",text="No image",fill=GREY,font=FSM,tags="ph")
            self._canvases[title]=c
            c.bind("<Configure>",lambda e,k=title:self._redraw(k))

    # ── Control bar (view toggle) ──────────────────────────────────────────────
    def _build_controlbar(self):
        bar=tk.Frame(self,bg=WHITE,height=46); bar.pack(fill="x",side="bottom")
        bar.pack_propagate(False)
        tk.Frame(bar,bg=BORDER,height=1).pack(fill="x")
        inner=tk.Frame(bar,bg=WHITE); inner.pack(side="left",padx=12,pady=6)
        tk.Label(inner,text="View Mode:",font=FSM,bg=WHITE,fg=GREY).pack(side="left",padx=(0,8))
        self._vbtns={}
        for v in VIEWS:
            b=tk.Button(inner,text=v,command=lambda mv=v:self._set_view(mv),
                        bg=BG,fg=DARK,relief="flat",font=FSM,padx=12,pady=5,
                        cursor="hand2",borderwidth=0)
            b.pack(side="left",padx=3)
            self._vbtns[v]=b
        self._set_view("Overlay")

        self._status_lbl=tk.Label(bar,text="Ready",font=FSM,bg=WHITE,fg=GREY)
        self._status_lbl.pack(side="right",padx=16)

    # ── Open picker ────────────────────────────────────────────────────────────
    def _open_picker(self):
        ImagePicker(self, self._on_image_selected)

    def _on_image_selected(self, path):
        self._img_path=path
        name=os.path.basename(path)
        self._status(f"Loaded: {name}")
        self._update_hist(path)
        if self.model: self._run()

    def _browse_model(self):
        p=filedialog.askopenfilename(title="Select weights",
            filetypes=[("PyTorch weights","*.pth"),("All","*.*")])
        if p: self._mpath.set(p); threading.Thread(target=self._load_model,daemon=True).start()

    # ── Model ──────────────────────────────────────────────────────────────────
    def _load_model(self):
        p=self._mpath.get()
        if not os.path.exists(p):
            self.after(0,self._mdl_lbl.config,{"text":"✘ Not found","fg":RED}); return
        try:
            m=ResUNet().to(self.device)
            m.load_state_dict(torch.load(p,map_location=self.device)); m.eval()
            self.model=m
            n=sum(pp.numel() for pp in m.parameters() if pp.requires_grad)
            self.after(0,lambda:self._mdl_lbl.config(
                text=f"✔ Ready · {n/1e6:.1f}M params",fg=GREEN))
        except Exception as ex:
            self.after(0,lambda:self._mdl_lbl.config(text=f"✘ {ex}",fg=RED))

    # ── Inference ──────────────────────────────────────────────────────────────
    def _run(self):
        if not self.model:
            messagebox.showwarning("Model not ready","Please wait for the model to load."); return
        if not self._img_path or not os.path.exists(self._img_path):
            messagebox.showwarning("No image","Click '📂 Open Image' first."); return
        self._status("⏳ Running inference…")
        threading.Thread(target=self._infer,daemon=True).start()

    def _infer(self):
        t0=time.time()
        try:
            bgr=cv2.imread(self._img_path)
            if bgr is None: raise ValueError("Cannot read image")
            thr=self._thr.get()
            rgb,mask,prob=run_model(bgr,self.model,self.device,thr)
            self._rgb=rgb; self._mask=mask; self._prob=prob
            ms=(time.time()-t0)*1000
            self.after(0,self._show,ms)
        except Exception as ex:
            self.after(0,self._status,f"✘ {ex}")

    def _show(self,ms):
        self._render_all()
        h,w=self._mask.shape; px=int(self._mask.sum()); ln=skel_len(self._mask)
        self._svars["px"].set(f"{px:,}")
        self._svars["cov"].set(f"{100*px/(h*w):.3f}%")
        self._svars["len"].set(f"~{ln:,} px")
        self._svars["ms"].set(f"{ms:.0f} ms")
        self._status(f"✔ Done — {px:,} seam pixels  ·  {ms:.0f} ms")

    def _render_all(self):
        rgb=self._rgb; mask=self._mask; prob=self._prob
        mid=self._build_mid(rgb,mask,prob)
        sk =make_skel(rgb,mask)
        for title,arr in [("Original",rgb),("Detection View",mid),("Skeleton / Centerline",sk)]:
            self._pil_cache[title]=np2pil(arr); self._redraw(title)

    def _build_mid(self,rgb,mask,prob):
        v=self._view.get()
        if   v=="Overlay":  return make_overlay(rgb,mask)
        elif v=="Mask":     return make_mask_only(mask)
        elif v=="Heat Map": return make_heat(prob,rgb)
        elif v=="Skeleton": return make_skel(rgb,mask)
        return make_overlay(rgb,mask)

    def _refresh(self):
        if self._rgb is None: return
        thr=self._thr.get()
        if self._prob is not None:
            self._mask=(self._prob>thr).astype(np.uint8)
        threading.Thread(target=self._render_all,daemon=True).start()

    def _set_view(self,mode):
        self._view.set(mode)
        for v,b in self._vbtns.items():
            b.config(bg=ACC if v==mode else BG,
                     fg=WHITE if v==mode else DARK)
        if self._rgb is not None:
            self._render_all()

    def _redraw(self,key):
        if key not in self._pil_cache: return
        pil=self._pil_cache[key]; c=self._canvases[key]
        cw=c.winfo_width() or 400; ch=c.winfo_height() or 400
        r=min(cw/pil.width,ch/pil.height)
        res=pil.resize((int(pil.width*r),int(pil.height*r)),Image.LANCZOS)
        ref=ImageTk.PhotoImage(res); self._img_refs[key]=ref
        c.delete("all"); c.create_image(cw//2,ch//2,anchor="center",image=ref)

    # ── History ────────────────────────────────────────────────────────────────
    def _update_hist(self,path):
        if path in self._hist: self._hist.remove(path)
        self._hist.append(path); self._hist=self._hist[-8:]
        for w in self._hist_frame.winfo_children(): w.destroy()
        for p in reversed(self._hist):
            n=os.path.basename(p)
            label=("▶ " if p==path else "  ")+(n[:16]+"…" if len(n)>16 else n)
            b=tk.Button(self._hist_frame,text=label,command=lambda p=p:self._on_image_selected(p),
                        bg=SEL if p==path else WHITE,fg=DARK,relief="flat",
                        font=FSM,anchor="w",padx=4,pady=3,cursor="hand2")
            b.pack(fill="x",pady=1)

    # ── Save ───────────────────────────────────────────────────────────────────
    def _save(self):
        if self._rgb is None:
            messagebox.showinfo("Nothing to save","Run inference first."); return
        p=filedialog.asksaveasfilename(defaultextension=".png",
                                       initialfile="seam_result.png",
                                       filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if not p: return
        ov=make_overlay(self._rgb,self._mask)
        sk=make_skel(self._rgb,self._mask)
        out=np.concatenate([self._rgb,ov,sk],axis=1)
        cv2.imwrite(p,cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Saved",f"3-panel result saved:\n{p}")

    # ── Misc ───────────────────────────────────────────────────────────────────
    def _status(self,msg): self._status_lbl.config(text=f"  {msg}")

    def _bind_keys(self):
        self.bind("<Return>",lambda e:self._run())
        self.bind("t",lambda e:self._cycle_view())
        self.bind("s",lambda e:self._save())
        self.bind("o",lambda e:self._open_picker())

    def _cycle_view(self):
        i=VIEWS.index(self._view.get()); self._set_view(VIEWS[(i+1)%len(VIEWS)])


if __name__=="__main__":
    App().mainloop()
