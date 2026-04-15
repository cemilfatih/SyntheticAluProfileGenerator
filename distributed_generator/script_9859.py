"""
Script 2/3: 9859 Profili (FAST)
Kaynak: Test Tuner - Package + Pattern + Tunnel Void
Optimizasyon: Vectorized texture, reduced depth steps, sprite caching
"""
import ezdxf
import cv2
import numpy as np
import os
import random
import math
import glob

# ==========================================
# ORTAK AYARLAR (3 scriptte de aynı olmalı)
# ==========================================
IMAGES_PER_CLASS_TRAIN = 2000
IMAGES_PER_CLASS_VAL = 50
IMG_WIDTH = 1280
IMG_HEIGHT = 1280
BASE_DIR = "dataset_5_profile_FINAL"
BG_FOLDER = "../background"
DXF_FOLDER = "../cad_files"

CLASS_ORDER = ['1524', '1560', '9859', '7170', '9794']
CLASS_MAP = {name: i for i, name in enumerate(CLASS_ORDER)}

DXF_PROFILE = os.path.join(DXF_FOLDER, "9859.dxf")
RENDER_SIZE = 150
PROFILES_PER_PKG = 6

# ==========================================
# TEST TUNER (9859 - FAST)
# ==========================================
class TestTuner:
    def __init__(self, bg_folder):
        self.bg_images = glob.glob(os.path.join(bg_folder, "*.*"))
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT

    def get_polygons_from_lines(self, lines, tolerance=3.0):
        edges = [[np.array(p1), np.array(p2)] for p1, p2 in lines]
        polygons = []
        while edges:
            current_poly = [edges[0][0], edges[0][1]]
            edges.pop(0)
            while True:
                last_pt = current_poly[-1]
                best_dist, best_idx, best_reverse = float('inf'), -1, False
                for i, edge in enumerate(edges):
                    d1 = np.linalg.norm(edge[0] - last_pt)
                    d2 = np.linalg.norm(edge[1] - last_pt)
                    if d1 < best_dist: best_dist, best_idx, best_reverse = d1, i, False
                    if d2 < best_dist: best_dist, best_idx, best_reverse = d2, i, True
                if best_idx != -1 and best_dist <= tolerance:
                    edge = edges.pop(best_idx)
                    current_poly.append(edge[0] if best_reverse else edge[1])
                else:
                    break
            if len(current_poly) > 2:
                polygons.append(np.array(current_poly, dtype=np.int32))
        return polygons

    def extract_mask_from_dxf(self, dxf_path, canvas_size):
        if not os.path.exists(dxf_path): return None
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        lines = []
        for e in msp:
            if e.dxftype() == 'LINE':
                lines.append(((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points('xy')
                for i in range(len(pts)-1): lines.append((pts[i], pts[i+1]))
                if e.closed: lines.append((pts[-1], pts[0]))
            elif e.dxftype() == 'ARC':
                c, r = e.dxf.center, e.dxf.radius
                sa, ea = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
                if ea < sa: ea += 2 * math.pi
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(sa, ea, 30)]
                for i in range(len(pts)-1): lines.append((pts[i], pts[i+1]))
            elif e.dxftype() == 'CIRCLE':
                c, r = e.dxf.center, e.dxf.radius
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 60)]
                for i in range(len(pts)-1): lines.append((pts[i], pts[i+1]))
        if not lines: return None
        all_pts = np.array([pt for p1, p2 in lines for pt in (p1, p2)])
        min_vals, max_vals = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
        max_dim = max(max_vals[0]-min_vals[0], max_vals[1]-min_vals[1])
        if max_dim == 0: max_dim = 1
        padding = int(canvas_size * 0.1)
        scale = (canvas_size - 2*padding) / max_dim
        scaled_lines = []
        for p1, p2 in lines:
            x1 = (p1[0]-min_vals[0])*scale+padding; y1 = canvas_size-((p1[1]-min_vals[1])*scale+padding)
            x2 = (p2[0]-min_vals[0])*scale+padding; y2 = canvas_size-((p2[1]-min_vals[1])*scale+padding)
            scaled_lines.append(((x1,y1),(x2,y2)))
        polygons = self.get_polygons_from_lines(scaled_lines, tolerance=3.0)
        polygons = sorted(polygons, key=cv2.contourArea, reverse=True)
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        for poly in polygons:
            temp = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            cv2.fillPoly(temp, [poly], 255)
            mask = cv2.bitwise_xor(mask, temp)
        return mask

    def apply_aluminum_texture(self, mask):
        h, w = mask.shape
        is_shiny = random.random() < 0.3
        base_val = random.randint(200, 240) if is_shiny else random.randint(150, 190)
        contrast = 30 if is_shiny else 15
        # FAST: vectorized gradient
        if random.random() < 0.5:
            gradient = np.clip(base_val - contrast * np.sin(np.arange(h, dtype=np.float32) / h * 6.28), 0, 255).astype(np.uint8)
            img_bgr = np.tile(gradient[:, np.newaxis, np.newaxis], (1, w, 3))
        else:
            gradient = np.clip(base_val - contrast * np.sin(np.arange(w, dtype=np.float32) / w * 6.28), 0, 255).astype(np.uint8)
            img_bgr = np.tile(gradient[np.newaxis, :, np.newaxis], (h, 1, 3))
        noise = np.random.randint(-12, 12, img_bgr.shape, dtype=np.int16)
        img_bgr = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours_cv, -1, (50, 50, 50), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    def get_single_profile(self, dxf_path, size):
        canvas_size = int(size * 3.0)
        mask = self.extract_mask_from_dxf(dxf_path, canvas_size)
        if mask is None: return None
        mask = cv2.GaussianBlur(mask, (3,3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None: return None
        x, y, w, h = cv2.boundingRect(coords)
        textured = self.apply_aluminum_texture(mask[y:y+h, x:x+w])
        if textured.shape[1] > textured.shape[0]:
            textured = cv2.rotate(textured, cv2.ROTATE_90_CLOCKWISE)
        return textured

    def create_package(self, sprite, n=6):
        h, w = sprite.shape[:2]
        pkg = np.zeros((h, w*n, 4), dtype=np.uint8)
        for i in range(n): pkg[:, i*w:(i+1)*w] = sprite
        return pkg

    def overlay_fast(self, img, overlay, px, py, alpha):
        hc, wc = img.shape[:2]
        oh, ow = overlay.shape[:2]
        px, py = int(px), int(py)
        x1, y1 = max(0,px), max(0,py)
        x2, y2 = min(wc,px+ow), min(hc,py+oh)
        if x2<=x1 or y2<=y1: return
        ox, oy = x1-px, y1-py
        hs, ws = y2-y1, x2-x1
        a = alpha[oy:oy+hs, ox:ox+ws].astype(np.float32)/255.0
        a3 = a[:,:,np.newaxis]
        fg = overlay[oy:oy+hs, ox:ox+ws, :3].astype(np.float32)
        bg = img[y1:y2, x1:x2].astype(np.float32)
        img[y1:y2, x1:x2] = (fg*a3 + bg*(1-a3)).astype(np.uint8)

    def generate_tunnel_void(self, w, h, ocx, ocy, cw, ch):
        vw, vh = w-4, h-4
        if vw <= 0 or vh <= 0: return None
        vx, vy = cw//2 - ocx, ch//2 - ocy
        lcx = np.clip(vw//2 + int(vx*0.3), 0, vw)
        lcy = np.clip(vh//2 + int(vy*0.3), 0, vh)
        Y, X = np.ogrid[:vh, :vw]
        dm = np.sqrt(((X-lcx)**2 + (Y-lcy)**2).astype(np.float32))
        md = np.sqrt(float(vw**2 + vh**2))
        nd = np.clip(dm / (md*0.6), 0, 1)
        alpha = (np.power(nd, 0.7) * random.uniform(0.80, 0.98) * 255).astype(np.uint8)
        base = np.zeros((vh, vw, 3), dtype=np.uint8)
        return np.dstack([base, alpha])

    def _queue_render(self, dl, fl, sprite, x, y, cw, ch):
        sh, sw = sprite.shape[:2]
        ocx, ocy = x+sw//2, y+sh//2
        vx, vy = cw//2 - ocx, ch//2 - ocy
        d = math.sqrt(vx**2+vy**2)
        md = math.sqrt((cw/2)**2+(ch/2)**2)
        dlen = min((d/md)*60, sw*0.8)
        void = self.generate_tunnel_void(sw, sh, ocx, ocy, cw, ch)
        dl.append({'x':x,'y':y,'sw':sw,'sh':sh,'alpha':sprite[:,:,3],'vx':vx,'vy':vy,'d':d,'dl':dlen,'void':void})
        fl.append({'sprite':sprite,'x':x,'y':y,'sw':sw,'sh':sh})

    def generate_pallet(self, dxf_path):
        cls_id = CLASS_MAP['9859']
        base_sprite = self.get_single_profile(dxf_path, RENDER_SIZE)
        if base_sprite is None: return None, []
        pkg_img = self.create_package(base_sprite, PROFILES_PER_PKG)
        pkg_h, pkg_w = pkg_img.shape[:2]
        
        pattern = random.choices(["ALTERNATING","SANDWICH","UNIFORM"], weights=[0.70,0.20,0.10], k=1)[0]
        num_rows = random.randint(8, 18)
        pkgs_per_row = random.randint(3, 6)
        pile_width = pkgs_per_row * pkg_w
        canvas_w = max(pile_width + random.randint(300,600), self.width)
        canvas_h = max((num_rows+2)*max(pkg_h,pkg_w)+200, self.height)

        # Background
        if self.bg_images and random.random() < 0.9:
            bg = cv2.imread(random.choice(self.bg_images))
            if bg is not None:
                wall = cv2.resize(bg, (canvas_w, canvas_h))
                wall = (wall.astype(np.float32)*random.uniform(0.5,0.95)).astype(np.uint8)
            else:
                wall = np.full((canvas_h, canvas_w, 3), random.randint(50,110), dtype=np.uint8)
        else:
            wall = np.full((canvas_h, canvas_w, 3), random.randint(50,110), dtype=np.uint8)
        noise = np.random.randint(-25, 25, wall.shape, dtype=np.int16)
        wall = np.clip(wall.astype(np.int16)+noise, 0, 255).astype(np.uint8)

        cur_y = canvas_h - random.randint(80,150)
        start_x = (canvas_w - pile_width)//2
        dl, fl = [], []

        for row_idx in range(num_rows):
            rtype = "HORIZONTAL"; flip = False
            if pattern == "ALTERNATING" and row_idx%2==1: flip = True
            elif pattern == "SANDWICH" and (num_rows//3 <= row_idx < 2*(num_rows//3)): rtype = "VERTICAL"
            if rtype == "HORIZONTAL":
                cur_y -= pkg_h
                if cur_y < 0: break
                for col in range(pkgs_per_row):
                    x = start_x + col*pkg_w
                    stamp = cv2.rotate(pkg_img, cv2.ROTATE_180) if flip else pkg_img
                    self._queue_render(dl, fl, stamp, x, cur_y, canvas_w, canvas_h)
            else:
                stamp = cv2.rotate(pkg_img, cv2.ROTATE_90_CLOCKWISE)
                cur_y -= stamp.shape[0]
                if cur_y < 0: break
                v_pkgs = pile_width // stamp.shape[1] if stamp.shape[1]>0 else 1
                off_x = start_x + (pile_width - v_pkgs*stamp.shape[1])//2
                for col in range(v_pkgs):
                    x = off_x + col*stamp.shape[1]
                    self._queue_render(dl, fl, stamp, x, cur_y, canvas_w, canvas_h)

        # Depth (reduced steps)
        for c in dl:
            if c['d'] > 0:
                nx_v, ny_v = c['vx']/c['d'], c['vy']/c['d']
                tox, toy = int(nx_v*c['dl']), int(ny_v*c['dl'])
                body = np.full((c['sh'],c['sw'],3), random.randint(30,60), dtype=np.uint8)
                steps = min(int(max(abs(tox),abs(toy))), 12)
                for s in range(steps, 0, -2):
                    r = s/steps if steps>0 else 0
                    self.overlay_fast(wall, body, c['x']+int(tox*r), c['y']+int(toy*r), c['alpha'])
            if c['void'] is not None:
                self.overlay_fast(wall, c['void'], c['x']+2, c['y']+2, c['void'][:,:,3])
        for c in fl:
            self.overlay_fast(wall, c['sprite'], c['x'], c['y'], c['sprite'][:,:,3])

        # Crop & resize to 1280x1280
        crop_top = max(0, cur_y - 150)
        final = cv2.resize(wall[crop_top:canvas_h, :], (self.width, self.height))

        # Annotations
        anns = []
        sx = self.width / wall.shape[1]
        sy = self.height / (canvas_h - crop_top)
        for c in fl:
            sw_c, sh_c = c['sw'], c['sh']
            px_c, py_c = c['x'], c['y'] - crop_top
            for i in range(PROFILES_PER_PKG):
                if sw_c > sh_c:
                    bw, bh = sw_c/float(PROFILES_PER_PKG), float(sh_c)
                    bx, by = px_c + i*bw, float(py_c)
                else:
                    bw, bh = float(sw_c), sh_c/float(PROFILES_PER_PKG)
                    bx, by = float(px_c), py_c + i*bh
                cx_n = ((bx+bw/2)*sx)/self.width
                cy_n = ((by+bh/2)*sy)/self.height
                if 0 <= cx_n <= 1 and 0 <= cy_n <= 1:
                    anns.append({'class_id': cls_id, 'bbox': [cx_n, cy_n, (bw*sx)/self.width, (bh*sy)/self.height]})
        return final, anns

def main():
    if not os.path.exists(DXF_PROFILE): print(f"❌ {DXF_PROFILE} bulunamadı!"); return
    tuner = TestTuner(bg_folder=BG_FOLDER)
    for s in ['train','val']:
        os.makedirs(f'{BASE_DIR}/images/{s}', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/labels/{s}', exist_ok=True)
    print(f"\n🚀 Script 2/3 FAST: 9859 Başlıyor...")
    for i in range(IMAGES_PER_CLASS_TRAIN):
        if i % 20 == 0: print(f"   [Train] {i}/{IMAGES_PER_CLASS_TRAIN}")
        p, a = tuner.generate_pallet(DXF_PROFILE)
        if p is None: continue
        n = f"9859_train_{i}"
        cv2.imwrite(f'{BASE_DIR}/images/train/{n}.jpg', p)
        ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
        with open(f'{BASE_DIR}/labels/train/{n}.txt', 'w') as f: f.write(ls)
    for i in range(IMAGES_PER_CLASS_VAL):
        if i % 5 == 0: print(f"   [Val] {i}/{IMAGES_PER_CLASS_VAL}")
        p, a = tuner.generate_pallet(DXF_PROFILE)
        if p is None: continue
        n = f"9859_val_{i}"
        cv2.imwrite(f'{BASE_DIR}/images/val/{n}.jpg', p)
        ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
        with open(f'{BASE_DIR}/labels/val/{n}.txt', 'w') as f: f.write(ls)
    print(f"\n✅ Script 2/3 TAMAMLANDI!")

if __name__ == "__main__":
    main()
