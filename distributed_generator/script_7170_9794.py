"""
Script 3/3: 7170 ve 9794 Profilleri (FAST)
Kaynak: V24 (PITCH BLACK TUNNEL) - Scattered + Tunnel Shadow
Optimizasyon: Vectorized texture, reduced depth steps, fast overlay
"""
import ezdxf
import cv2
import numpy as np
import random
import os
import math
import glob
import networkx as nx

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

MY_PROFILES = {
    '7170': os.path.join(DXF_FOLDER, "7170.dxf"),
    '9794': os.path.join(DXF_FOLDER, "9794.dxf"),
}

# ==========================================
# DXF YÜKLEYİCİ (V24 - Single Contour)
# ==========================================
class DXFProfileLoader:
    def __init__(self):
        self.profiles = {}

    def load_dxf(self, dxf_path, profile_name):
        if not os.path.exists(dxf_path):
            print(f"⚠️  {dxf_path} bulunamadı!")
            return None
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            G = nx.Graph()
            def to_key(p): return (round(p[0], 3), round(p[1], 3))
            has_data = False
            for e in msp:
                if e.dxftype() == 'LINE':
                    G.add_edge(to_key(e.dxf.start), to_key(e.dxf.end)); has_data = True
                elif e.dxftype() == 'LWPOLYLINE':
                    pts = e.get_points('xy')
                    for i in range(len(pts)-1): G.add_edge(to_key(pts[i]), to_key(pts[i+1]))
                    if e.closed: G.add_edge(to_key(pts[-1]), to_key(pts[0]))
                    has_data = True
            if not has_data: return None
            components = list(nx.connected_components(G))
            largest = max(components, key=len)
            subgraph = G.subgraph(largest)
            ordered = list(nx.dfs_preorder_nodes(subgraph))
            pts = np.array(ordered)
            min_v, max_v = np.min(pts, axis=0), np.max(pts, axis=0)
            center = (min_v + max_v) / 2
            pts = pts - center
            h = max_v[1] - min_v[1]
            self.profiles[profile_name] = pts / h if h > 0 else pts
            print(f"✅ Yüklendi: {profile_name}")
            return self.profiles[profile_name]
        except Exception as e:
            print(f"❌ DXF Hatası ({profile_name}): {e}")
            return None

# ==========================================
# RENDER MOTORU (V24 - FAST)
# ==========================================
class ProfileRenderer:
    def __init__(self): pass

    def apply_aluminum_texture(self, mask):
        h, w = mask.shape
        is_shiny = random.random() < 0.30
        if is_shiny:
            base_val, contrast = random.randint(230, 255), 40
        else:
            base_val, contrast = random.randint(180, 220), 25
        # FAST: vectorized
        if random.random() < 0.5:
            grad = np.clip(base_val - contrast * np.sin(np.arange(h, dtype=np.float32) / h * 3.14), 0, 255).astype(np.uint8)
            img_bgr = np.tile(grad[:, np.newaxis, np.newaxis], (1, w, 3))
        else:
            grad = np.clip(base_val - contrast * np.sin(np.arange(w, dtype=np.float32) / w * 3.14), 0, 255).astype(np.uint8)
            img_bgr = np.tile(grad[np.newaxis, :, np.newaxis], (h, 1, 3))
        noise = np.random.randint(-15, 15, img_bgr.shape, dtype=np.int16)
        img_bgr = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours_cv, -1, (60, 60, 60), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    def get_single_half(self, contour, size, rotation=0):
        canvas_size = int(size * 2.5)
        scaled = contour * size + canvas_size // 2
        if rotation != 0:
            M = cv2.getRotationMatrix2D((canvas_size//2, canvas_size//2), rotation, 1.0)
            ones = np.ones((len(scaled), 1))
            scaled = M.dot(np.hstack([scaled, ones]).T).T
        pts = scaled.astype(np.int32)
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None: return None, 0
        x, y, w, h = cv2.boundingRect(coords)
        thickness = int(min(w, h) * 0.15)
        return self.apply_aluminum_texture(mask[y:y+h, x:x+w]), thickness

# ==========================================
# PALET ÜRETİCİ (V24 - FAST)
# ==========================================
class SyntheticPalletGenerator:
    def __init__(self, loader, renderer, bg_folder='background'):
        self.loader = loader
        self.renderer = renderer
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.bg_images = glob.glob(os.path.join(bg_folder, "*.*"))

    def get_random_background(self):
        if self.bg_images and random.random() < 0.95:
            bg = cv2.imread(random.choice(self.bg_images))
            if bg is not None:
                bg = cv2.resize(bg, (self.width, self.height))
                return (bg.astype(np.float32) * random.uniform(0.5, 0.9)).astype(np.uint8)
        val = random.randint(40, 100)
        bg = np.full((self.height, self.width, 3), val, dtype=np.uint8)
        noise = np.random.randint(-30, 30, bg.shape, dtype=np.int16)
        return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def generate_tunnel_void(self, w, h, ocx, ocy, thickness):
        vw, vh = w - 2*thickness, h - 2*thickness
        if vw <= 0 or vh <= 0: return None
        vx, vy = self.width//2 - ocx, self.height//2 - ocy
        lcx = np.clip(vw//2 + int(vx*0.3), 0, vw)
        lcy = np.clip(vh//2 + int(vy*0.3), 0, vh)
        Y, X = np.ogrid[:vh, :vw]
        dm = np.sqrt(((X-lcx)**2 + (Y-lcy)**2).astype(np.float32))
        md = np.sqrt(float(vw**2+vh**2))
        nd = np.clip(dm / (md*0.6), 0, 1)
        opacity = random.uniform(0.80, 0.95)
        alpha = (np.power(nd, 0.7) * opacity * 255).astype(np.uint8)
        base = np.random.randint(0, 20, (vh, vw, 3), dtype=np.uint8)
        return np.dstack([base, alpha])

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

    def generate_pallet(self, ptype):
        pallet = self.get_random_background()
        anns = []
        sf = self.width / 640.0
        contour = self.loader.profiles[ptype]
        bsz = int(random.randint(25, 40) * sf)
        spacer_r = (int(5*sf), int(15*sf))
        gap_r = (int(2*sf), int(5*sf))

        ref, ref_t = self.renderer.get_single_half(contour, bsz, 0)
        if ref is None: return pallet, []
        ph, pw = ref.shape[:2]
        if ph == 0 or pw == 0: return pallet, []

        cur_y = self.height - int(10*sf)
        depth_cmds, face_cmds = [], []

        while cur_y > int(50*sf):
            spacer = random.randint(*spacer_r)
            row_y = cur_y - spacer - ph
            if row_y < 0: break
            cur_x = random.randint(0, int(30*sf))
            while cur_x < self.width - pw:
                rot = random.choice([0, 180])
                sz = int(bsz * random.uniform(0.95, 1.05))
                sprite, t = self.renderer.get_single_half(contour, sz, rot)
                if sprite is None: break
                sh, sw = sprite.shape[:2]
                px, py = cur_x, row_y + random.randint(-2, 2)
                if px + sw > self.width: break
                ocx, ocy = px+sw//2, py+sh//2
                vx, vy = self.width//2 - ocx, self.height//2 - ocy
                d = math.sqrt(vx**2+vy**2)
                md = math.sqrt((self.width/2)**2+(self.height/2)**2)
                dl = min((d/md)*(50*sf), sw*0.8)
                # Tunnel void (V24)
                void = self.generate_tunnel_void(sw, sh, ocx, ocy, t)
                if void is not None:
                    self.overlay_fast(pallet, void, px+t, py+t, void[:,:,3])
                depth_cmds.append({'x':px,'y':py,'sw':sw,'sh':sh,'vx':vx,'vy':vy,'d':d,'dl':dl,'alpha':sprite[:,:,3]})
                face_cmds.append({'sprite':sprite,'x':px,'y':py,'sw':sw,'sh':sh})
                cur_x += sw + random.randint(*gap_r)
            cur_y = row_y

        # Depth (reduced: max 15 steps, step=2)
        for c in depth_cmds:
            if c['d'] <= 0: continue
            nx_v, ny_v = c['vx']/c['d'], c['vy']/c['d']
            tox, toy = int(nx_v*c['dl']), int(ny_v*c['dl'])
            body = np.full((c['sh'], c['sw'], 3), 50, dtype=np.uint8)
            steps = min(int(max(abs(tox), abs(toy))), 15)
            for s in range(steps, 0, -2):
                r = s/steps if steps>0 else 0
                self.overlay_fast(pallet, body, c['x']+int(tox*r), c['y']+int(toy*r), c['alpha'])

        cls_id = CLASS_MAP[ptype]
        for c in face_cmds:
            self.overlay_fast(pallet, c['sprite'], c['x'], c['y'], c['sprite'][:,:,3])
            anns.append({'class_id': cls_id, 'bbox': [
                (c['x']+c['sw']/2)/self.width, (c['y']+c['sh']/2)/self.height,
                c['sw']/self.width, c['sh']/self.height]})

        if random.random() < 0.3:
            noise = np.random.normal(0, 3, pallet.shape).astype(np.int16)
            pallet = np.clip(pallet.astype(np.int16)+noise, 0, 255).astype(np.uint8)
        return pallet, anns

def main():
    loader = DXFProfileLoader()
    for name, path in MY_PROFILES.items(): loader.load_dxf(path, name)
    if not loader.profiles: print("❌ Profil yüklenemedi!"); return
    renderer = ProfileRenderer()
    gen = SyntheticPalletGenerator(loader, renderer, BG_FOLDER)
    for s in ['train','val']:
        os.makedirs(f'{BASE_DIR}/images/{s}', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/labels/{s}', exist_ok=True)
    print(f"\n🚀 Script 3/3 FAST: 7170 + 9794 Başlıyor...")
    for ptype in MY_PROFILES:
        if ptype not in loader.profiles: continue
        print(f"\n📦 {ptype} ...")
        for i in range(IMAGES_PER_CLASS_TRAIN):
            if i % 20 == 0: print(f"   [Train] {i}/{IMAGES_PER_CLASS_TRAIN}")
            p, a = gen.generate_pallet(ptype)
            n = f"{ptype}_train_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/train/{n}.jpg', p)
            ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
            with open(f'{BASE_DIR}/labels/train/{n}.txt', 'w') as f: f.write(ls)
        for i in range(IMAGES_PER_CLASS_VAL):
            if i % 5 == 0: print(f"   [Val] {i}/{IMAGES_PER_CLASS_VAL}")
            p, a = gen.generate_pallet(ptype)
            n = f"{ptype}_val_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/val/{n}.jpg', p)
            ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
            with open(f'{BASE_DIR}/labels/val/{n}.txt', 'w') as f: f.write(ls)
    print(f"\n✅ Script 3/3 TAMAMLANDI!")

if __name__ == "__main__":
    main()
