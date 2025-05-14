import os
import cv2
import json
import yaml
import torch
import shutil
import argparse
import numpy as np
import tkinter as tk

from utils import util
from pathlib import Path
from ttkbootstrap import Style
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from ttkbootstrap.widgets import Button, Frame, Label
from ttkbootstrap.widgets import Labelframe, PanedWindow, Scrollbar


class FaceLandmarkAutoLabeler:
    def __init__(self,
                 cfg="utils/cfg.yaml",
                 det="weights/detection.onnx",
                 lmk="weights/best.pt"):

        self.params = yaml.safe_load(open(cfg))
        self.detector = util.FaceDetector(det)
        self.lmk_model = torch.load(lmk)["model"]
        self.lmk_model = self.lmk_model.float().cuda().eval()

        self.mean = np.array([58.395, 57.12, 57.375], np.float64).reshape(1,
                                                                          -1)
        self.std = self.mean.copy()

        self.dataset = {"images": [], "landmarks": [],
                        "metadata": {"num_landmarks": self.params["num_lms"],
                                     "date_created": None, "total_samples": 0}}

        self.root = None
        self.canvas = None
        self.hbar = None
        self.vbar = None
        self._grab = None
        self.edit_btn = None
        self.is_editing = False
        self.current_path = None
        self.current_image = None
        self.current_landmarks = None

        self._zoom = 1.0
        self._scale = 1.0
        self.scale = 1.2
        self.current_idx = -1
        self.landmark_radius = 2

    # ────────────────────────────  UI  ────────────────────────────
    def build_ui(self):
        style = Style(theme="flatly")
        self.root = style.master
        self.root.title("Face Landmark Auto-Labeling")
        self.root.geometry("1280x820")
        style.configure(".", font=("Segoe UI", 10))
        self._build_menu()
        self._build_bar()
        self._build_panes()
        self._bind_events()

    def _build_menu(self):
        # def _build_menu(self):
        menubar = tk.Menu(self.root)
        f = tk.Menu(menubar, tearoff=False)
        f.add_command(label="Open image…   Ctrl+O", command=self.open_image)
        f.add_command(label="Open folder… Ctrl+Shift+O", command=self.open_dir)
        f.add_separator()
        f.add_command(label="Save sample   Ctrl+S", command=self.save_current)
        f.add_separator()
        f.add_command(label="Export…    Ctrl+E", command=self.export_dataset)
        f.add_separator()
        f.add_command(label="Exit", command=self.safe_exit)  # ← update
        menubar.add_cascade(label="File", menu=f)

        theme = tk.Menu(menubar, tearoff=False)
        for th in Style(theme="flatly").theme_names():
            theme.add_radiobutton(label=th, command=lambda t=th: Style()
                                  .theme_use(t))
        menubar.add_cascade(label="Theme", menu=theme)
        self.root.config(menu=menubar)

    def _build_bar(self):
        bar = Frame(self.root, bootstyle="secondary")
        bar.pack(fill=tk.X, ipadx=4)

        def add(txt, cmd):
            b = Button(bar, text=txt, width=10,
                       bootstyle="outline-secondary", command=cmd)
            b.pack(side=tk.LEFT, padx=2, pady=2)
            return b

        for txt, cmd in [("Image", self.open_image),
                         ("Folder", self.open_dir),
                         ("Detect", self.auto_label_current),
                         ("Edit", self.toggle_edit_mode),
                         ("Save", self.save_current),
                         ("Export", self.export_dataset),
                         ("Prev", self.prev_image),
                         ("Next", self.next_image)]:
            btn = add(txt, cmd)
            if txt == "Edit": self.edit_btn = btn
        return bar

    def _build_panes(self):
        paned = PanedWindow(self.root, orient=tk.HORIZONTAL,
                            bootstyle="secondary")
        paned.pack(fill=tk.BOTH, expand=True)

        lframe = Frame(paned)
        self.hbar = Scrollbar(lframe, orient=tk.HORIZONTAL,
                              command=lambda *a: self.canvas.xview(*a),
                              bootstyle="secondary-round")
        self.vbar = Scrollbar(lframe, orient=tk.VERTICAL,
                              command=lambda *a: self.canvas.yview(*a),
                              bootstyle="secondary-round")
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas = tk.Canvas(lframe, bg="#2b2b2b", highlightthickness=0,
                                xscrollcommand=self.hbar.set,
                                yscrollcommand=self.vbar.set)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        paned.add(lframe, weight=4)

        # info panel
        rframe = Frame(paned, width=260)
        info = Labelframe(rframe, text="Information")
        stats = Labelframe(rframe, text="Dataset stats")
        self.info_txt = tk.Text(info, height=8, width=32, wrap="none")
        self.stats_txt = tk.Text(stats, height=4, width=32, wrap="none")
        self.info_txt = tk.Text(info, height=8, width=32, wrap="none",
                                state="disabled", bg=Style().colors.light)
        self.stats_txt = tk.Text(stats, height=4, width=32, wrap="none",
                                 state="disabled", bg=Style().colors.light)
        self.info_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.stats_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        info.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 3))
        stats.pack(fill=tk.BOTH, expand=False, padx=6, pady=(3, 6))
        paned.add(rframe, weight=0)

        # status bar
        self.status = tk.StringVar(value="Ready")
        Label(self.root, textvariable=self.status, anchor=tk.W,
              bootstyle="inverse-light").pack(fill=tk.X)

    def _bind_events(self):
        # ───── key / mouse bindings
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-O>", lambda e: self.open_dir())
        self.root.bind("<Control-s>", lambda e: self.save_current())
        self.root.bind("<Control-e>", lambda e: self.export_dataset())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        # ★ QUIT bindings
        self.root.bind("<Escape>", lambda e: self.safe_exit())
        self.root.bind("<KeyPress-q>", lambda e: self.safe_exit())
        self.root.bind("<KeyPress-Q>", lambda e: self.safe_exit())

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)

    def open_image(self):
        fmt = "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"
        p = filedialog.askopenfilename(filetypes=
                                       [("Images", fmt), ("All", "*.*")])
        if p: self.load_image(p)

    def show_image(self, idx):
        self._zoom = 1.0
        img = self.dataset["images"]
        lmk = self.dataset["landmarks"]
        if not (0 <= idx < len(img)): return

        rec = img[idx]
        self.current_idx, self.current_path = idx, rec["path"]
        self.current_image = util.imread_unicode(rec["path"])
        self.current_landmarks = lmk[idx]

        self.redraw()
        self.write_info(
            f"File: {rec['filename']}\nSize: {rec['width']}×{rec['height']}")
        self.status.set(f"{idx + 1}/{len(self.dataset['images'])}" + (
            "   EDITING ON" if self.is_editing else ""))

    def switch_image(self, step):
        if self.dataset["images"]:
            self.show_image(
                (self.current_idx + step) % len(self.dataset["images"]))

    def next_image(self):
        self.switch_image(1)

    def prev_image(self):
        self.switch_image(-1)

    def open_dir(self):
        if not (folder := Path(
                filedialog.askdirectory(title="Select image folder") or "")):
            return  # user cancelled

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
        files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        if not files:
            return messagebox.showinfo("No images", folder)

        self.status.set(f"Batch-processing {len(files)} images…")
        self.root.update_idletasks()

        for f in files:  # one compact pass
            self.load_image(f)
            self.auto_label_current()
            self.current_landmarks and self.save_current()

        self.status.set("Batch complete")
        self.update_stats()
        self.dataset["images"] and self.show_image(0)

    def load_image(self, path):
        if (
                img := util.imread_unicode(
                    path)) is None:  # read-and-check in one go
            return messagebox.showerror("Load error", "Cannot read image")

        self.current_path, self.current_image, self.current_landmarks = path, img, None
        self._zoom = 1.0
        self.redraw()

        h, w = img.shape[:2]
        self.write_info(f"File: {Path(path).name}\nSize: {w}×{h}")

    def on_zoom(self, event):
        if not event.state & 0x0004:
            return
        factor = 1.1 if (event.num == 4 or event.delta > 0) else 0.9
        newzoom = max(.2, min(5.0, self._zoom * factor))

        if abs(newzoom - self._zoom) < 1e-3:
            return

        self._zoom = newzoom
        self.redraw()

    def redraw(self):
        if (img := self.current_image) is None:
            return

        cW, cH = self.canvas.winfo_width(), self.canvas.winfo_height()
        ih, iw = img.shape[:2]
        self._scale = min(cW / iw, cH / ih) if cW > 10 and cH > 10 else 1
        eff = self._scale * self._zoom

        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        show = cv2.resize(rgb,
                          (int(iw * eff), int(ih * eff))) if eff != 1 else rgb

        self.tkimg = ImageTk.PhotoImage(Image.fromarray(show))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tkimg, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, show.shape[1], show.shape[0]))

        if self.current_landmarks: self.draw_landmarks()

    def draw_landmarks(self):
        s = self._scale * self._zoom
        edit = self.is_editing
        c_oval = self.canvas.create_oval
        c_rect = self.canvas.create_rectangle

        box_col, box_w, dot_r, dot_col = ("#FF4D4F", 3, 6, "#FF9500") if (
            edit) else ("#00A3FF", 1, 2, "#25FF00")

        for face in self.current_landmarks:
            x0, y0, x1, y1 = [v * s for v in face["bbox"]]
            c_rect(x0, y0, x1, y1, outline=box_col, width=box_w)

            for (x, y) in face["landmarks"]:
                xs, ys = x * s, y * s
                c_oval(xs - dot_r, ys - dot_r, xs + dot_r, ys + dot_r,
                       fill=dot_col, outline="")

    # ────────────────────────────  Editing  ──────────────────────────────────
    def on_click(self, ev):
        if not (self.is_editing and self.current_landmarks): return
        ix, iy = self.canvas_to_img(ev.x, ev.y)
        best = (10 ** 9, None, None)
        for fi, face in enumerate(self.current_landmarks):
            for li, (x, y) in enumerate(face["landmarks"]):
                d = ((ix - x) ** 2 + (iy - y) ** 2) ** 0.5
                if d < best[0] and d < 12: best = (d, fi, li)
        if best[1] is not None:
            self._grab = (best[1], best[2])
            self.canvas.configure(cursor="fleur")  # ★ NEW
            self.status.set("Drag to reposition")

    def on_drag(self, ev):
        c_img = self.canvas_to_img
        c_lmk = self.current_landmarks
        if self._grab is None: return
        fi, li = self._grab
        c_lmk[fi]["landmarks"][li] = c_img(ev.x, ev.y)
        self.redraw()

    def on_release(self, _):
        self._grab = None
        self.canvas.configure(cursor="")  # ★ NEW
        if self.is_editing: self.status.set("EDITING ON")

    def canvas_to_img(self, cx, cy):
        # use canvas coords (with scrolling)
        sx = self.canvas.canvasx(cx)
        sy = self.canvas.canvasy(cy)
        s = self._scale * self._zoom
        return int(sx / s), int(sy / s)

    def toggle_edit_mode(self):
        self.is_editing = not self.is_editing

        if self.edit_btn:
            new_style = "danger" if self.is_editing else "outline-secondary"
            self.edit_btn.configure(bootstyle=new_style)

        glow_col = "#FF4D4F" if self.is_editing else "#2b2b2b"
        self.canvas.configure(highlightthickness=4 if self.is_editing else 0,
                              highlightbackground=glow_col)

        # landmark visual params
        self.landmark_radius = 6 if self.is_editing else 2
        self.redraw()

        self.status.set("EDITING ON" if self.is_editing else "View mode")

    # ────────────────────────────  Landmark ops  ─────────────────────────────
    def auto_label_current(self):
        if self.current_image is None: return
        lms = self.detect(self.current_image)
        if not lms: messagebox.showinfo("No face", "None found"); return
        self.current_landmarks = lms
        self.redraw()
        self.status.set(f"{len(lms)} face(s) detected")

    def detect(self, img):
        boxes = self.detector.detect(img, (640, 640))
        if boxes is None or len(boxes) == 0: return None
        result = []
        for b in boxes:
            proc, _box, _xy = util.process_face(b, img, self.scale,
                                                self.params, self.mean,
                                                self.std)
            with torch.no_grad():
                out = self.lmk_model(proc)
            if isinstance(out, torch.Tensor):
                pts, conf  = out, torch.ones(1)
            elif isinstance(out, (tuple, list)):
                pts, conf = (out + [torch.ones(1)])[:2]
            else:
                pts, conf = out.get("landmarks") or next(
                    iter(out.values())), out.get("scores") or torch.ones(1)
            pts = pts.detach().cpu().numpy().flatten()
            face = []
            for i in range(self.params["num_lms"]):
                x = int(pts[2 * i] * _box[0]) + _xy[0];
                y = int(pts[2 * i + 1] * _box[1]) + _xy[1]
                face.append((x, y))
            result.append({"bbox": [*_xy], "landmarks": face,
                           "confidence": float(conf.mean())})
        return result

    def save_current(self):
        if not (self.current_path and self.current_landmarks): return
        name = os.path.basename(self.current_path)
        entry = {"filename": name, "path": self.current_path,
                 "width": self.current_image.shape[1],
                 "height": self.current_image.shape[0]}
        imgs, lms = self.dataset["images"], self.dataset["landmarks"]
        try:
            idx = next(i for i, e in enumerate(imgs) if e["filename"] == name)
        except StopIteration:
            idx = -1
        if idx >= 0:
            imgs[idx] = entry;
            lms[idx] = self.current_landmarks
        else:
            imgs.append(entry);
            lms.append(self.current_landmarks)
        self.update_stats();
        self.status.set("Sample saved")

    # ────────────────────────────  Export  ───────────────────────────────────
    def export_dataset(self):
        if not self.dataset["images"]:
            messagebox.showinfo("Empty", "Nothing to export")
            return

        export_dir = filedialog.askdirectory(
            parent=self.root,
            title="Choose export destination"
        )
        if not export_dir: return

        try:
            img_dir = os.path.join(export_dir, "images")
            lbl_dir = os.path.join(export_dir, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)

            for img_rec, lms_list in zip(self.dataset["images"],
                                         self.dataset["landmarks"]):
                if not lms_list:
                    continue

                dst_img = os.path.join(img_dir, img_rec["filename"])
                if not os.path.exists(dst_img):
                    shutil.copy2(img_rec["path"], dst_img)

                anno = {
                    "filename": img_rec["filename"],
                    "width": img_rec["width"],
                    "height": img_rec["height"],
                    "faces": self._faces_json(lms_list),
                    "num_landmarks": self.params["num_lms"],
                }
                json_name = os.path.splitext(img_rec["filename"])[0] + ".json"
                with open(os.path.join(lbl_dir, json_name), "w") as jf:
                    json.dump(anno, jf, indent=2)

                for idx, face in enumerate(lms_list):
                    # construct output name:  <image>_f0.pts, _f1.pts … if >1 face
                    base = os.path.splitext(img_rec["filename"])[0]
                    pts_name = f"{base}.pts" if len(lms_list) == 1 \
                        else f"{base}_f{idx}.pts"
                    with open(os.path.join(lbl_dir, pts_name), "w") as pf:
                        for (x, y) in face["landmarks"][
                                      :self.params["num_lms"]]:
                            pf.write(f"{int(x)} {int(y)}\n")

            messagebox.showinfo(
                "Export complete",
                f"Images copied to {img_dir}\n"
                f"JSON & PTS files saved to {lbl_dir}"
            )

        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _faces_json(self, faces):
        return [{"bbox": [int(v) for v in f["bbox"]],
                 "points": [[int(x), int(y)] for (x, y) in f["landmarks"]],
                 "confidence": float(f["confidence"])} for f in faces]

    # ────────────────────────────  Misc helpers  ────────────────────────────
    def write_info(self, txt):
        self.info_txt.configure(state="normal")
        self.info_txt.delete(1.0, tk.END)
        self.info_txt.insert(tk.END, txt)
        self.info_txt.configure(state="disabled")

    def update_stats(self):
        self.dataset["metadata"]["total_samples"] = len(self.dataset["images"])
        faces = sum(len(l) for l in self.dataset["landmarks"] if l)
        pts = sum(
            len(f["landmarks"]) for l in self.dataset["landmarks"] if l for f
            in l)
        self.stats_txt.configure(state="normal")
        self.stats_txt.delete(1.0, tk.END)
        self.stats_txt.insert(tk.END,
                              f"Images: {len(self.dataset['images'])}\nFaces: {faces}\nLandmarks: {pts}\n")
        self.stats_txt.configure(state="disabled")

    # ────────────────────────────  Run  ────────────────────────────
    def run(self):
        self.build_ui()
        self.root.mainloop()

    def safe_exit(self):
        if messagebox.askokcancel("Quit", "Exit the program?"):
            self.root.destroy()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Face Landmark Auto-Labeling")
    ap.add_argument("--config", default="utils/cfg.yaml")
    ap.add_argument("--detector", default="weights/detection.onnx")
    ap.add_argument("--landmark", default="weights/best.pt")
    args = ap.parse_args()
    FaceLandmarkAutoLabeler(args.config, args.detector, args.landmark).run()


if __name__ == "__main__":
    main()
