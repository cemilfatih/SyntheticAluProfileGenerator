"""
Script 1/3: 1524 ve 1560 Profilleri (FAST)
Kaynak: V14 (FIXED GEOMETRY) - Paired Block + Void Fill
Optimizasyon: Vectorized NumPy, Sprite Caching, Reduced Depth Steps
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
    '1524': os.path.join(DXF_FOLDER, "1524_CLEAN.dxf"),
    '1560': os.path.join(DXF_FOLDER, "1560_CLEAN.dxf"),
}

SPECIAL_PAIRS_CONFIG = {
    '1524': 0.70,
    '1560': 0.85,
}

# ==========================================
# DXF YÜKLEYİCİ (V14 - Multi Contour Graph)
# ==========================================
class DXFProfileLoader:
    def arc_to_points(self, entity, s=30):
        c, r = entity.dxf.center, entity.dxf.radius
        start = math.radians(entity.dxf.start_angle)
        end = math.radians(entity.dxf.end_angle)
        if end < start: end += 2 * math.pi
        return [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(start, end, s)]

    def circle_to_points(self, entity, s=60):
        c, r = entity.dxf.center, entity.dxf.radius
        return [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, s)]

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
                elif e.dxftype() == 'ARC':
                    pts = self.arc_to_points(e)
                    for i in range(len(pts)-1): G.add_edge(to_key(pts[i]), to_key(pts[i+1]))
                    has_data = True
                elif e.dxftype() == 'CIRCLE':
                    pts = self.circle_to_points(e)
                    for i in range(len(pts)-1): G.add_edge(to_key(pts[i]), to_key(pts[i+1]))
                    G.add_edge(to_key(pts[-1]), to_key(pts[0]))
                    has_data = True
            if not has_data: return None
            components = list(nx.connected_components(G))
            main_comp = max(components, key=len)
            main_pts = np.array(list(main_comp))
            main_min, main_max = np.min(main_pts, axis=0), np.max(main_pts, axis=0)
            valid_comps = []
            for comp in components:
                if len(comp) < 3: continue
                pts = np.array(list(comp))
                c_min, c_max = np.min(pts, axis=0), np.max(pts, axis=0)
                if comp == main_comp or (np.all(c_min >= main_min - 0.1) and np.all(c_max <= main_max + 0.1)):
                    valid_comps.append(comp)
            if not valid_comps: return None
            contours = []
            all_pts = []
            for comp in valid_comps:
                subgraph = G.subgraph(comp)
                start_node = next((n for n, d in subgraph.degree() if d == 1), list(comp)[0])
                ordered_nodes = list(nx.dfs_preorder_nodes(subgraph, source=start_node))
                contours.append(np.array(ordered_nodes))
                all_pts.extend(ordered_nodes)
            all_pts = np.array(all_pts)
            min_vals, max_vals = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
            center = (min_vals + max_vals) / 2
            max_dim = np.max(max_vals - min_vals)
            normalized_contours = []
            for pts in contours:
                pts = pts - center
                if max_dim > 0: pts = pts / max_dim
                normalized_contours.append(pts)
            self.profiles[profile_name] = normalized_contours
            print(f"✅ Yüklendi: {profile_name} ({len(normalized_contours)} contour)")
            return normalized_contours
        except Exception as e:
            print(f"❌ DXF Hatası ({profile_name}): {e}")
            return None

# ==========================================
# RENDER MOTORU (V14 - FAST VECTORIZED)
# ==========================================
class ProfileRenderer:
    def __init__(self): pass

    def apply_aluminum_texture(self, mask):
        h, w = mask.shape
        base_val = random.randint(180, 220)
        # FAST: vectorized gradient
        if random.random() < 0.5:
            gradient = np.clip(base_val - 25 * np.sin(np.arange(h, dtype=np.float32) / h * 3.14), 0, 255).astype(np.uint8)
            img_bgr = np.tile(gradient[:, np.newaxis, np.newaxis], (1, w, 3))
        else:
            gradient = np.clip(base_val - 25 * np.sin(np.arange(w, dtype=np.float32) / w * 3.14), 0, 255).astype(np.uint8)
            img_bgr = np.tile(gradient[np.newaxis, :, np.newaxis], (h, 1, 3))
        noise = np.random.randint(-15, 15, img_bgr.shape, dtype=np.int16)
        img_bgr = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours_cv, -1, (60, 60, 60), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    def get_single_half(self, contours, size, rotation=0):
        canvas_size = int(size * 2.5)
        pts_list = []
        for contour in contours:
            scaled = contour * size + canvas_size // 2
            if rotation != 0:
                M = cv2.getRotationMatrix2D((canvas_size//2, canvas_size//2), rotation, 1.0)
                ones = np.ones((len(scaled), 1))
                scaled = M.dot(np.hstack([scaled, ones]).T).T
            pts_list.append(scaled.astype(np.int32))
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(mask, pts_list, 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None: return None, 0
        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y+h, x:x+w]
        thickness = int(w * 0.12)
        return self.apply_aluminum_texture(cropped_mask), thickness

    def create_paired_block(self, contours, size, profile_name, target_rotation=0):
        sprite_a, t = self.get_single_half(contours, size, 0)
        if sprite_a is None: return np.zeros((10,10,4), dtype=np.uint8), (0,0,0,0)
        h, w = sprite_a.shape[:2]
        sprite_b_raw, _ = self.get_single_half(contours, size, 180)
        sprite_b = cv2.resize(sprite_b_raw, (w, h))
        tightness = SPECIAL_PAIRS_CONFIG.get(profile_name, 1.0)
        shift_val = int(t * tightness)
        canvas_w = w * 2 + shift_val * 4
        canvas_h = h * 2 + shift_val * 4
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        cx, cy = canvas_w // 2, canvas_h // 2
        pos_a_x, pos_a_y = cx - w // 2, cy - h // 2
        pos_b_x, pos_b_y = pos_a_x + shift_val, pos_a_y + shift_val

        # VOID FILLING - FULLY VECTORIZED
        void_x1 = min(pos_a_x, pos_b_x) + t
        void_y1 = min(pos_a_y, pos_b_y) + t
        void_x2 = max(pos_a_x + w, pos_b_x + w) - t
        void_y2 = max(pos_a_y + h, pos_b_y + h) - t
        if void_x2 > void_x1 and void_y2 > void_y1:
            vw, vh = void_x2 - void_x1, void_y2 - void_y1
            if profile_name == '1560':
                base_val = np.random.randint(50, 90, (vh, vw), dtype=np.uint8)
            else:
                base_val = np.random.randint(10, 30, (vh, vw), dtype=np.uint8)
            void_img_bgr = np.stack([base_val]*3, axis=-1)
            noise = np.random.randint(-20, 20, (vh, vw, 3), dtype=np.int16)
            void_img_bgr = np.clip(void_img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # Vectorized vignette
            Y, X = np.ogrid[:vh, :vw]
            dist_map = np.sqrt(((X - vw//2)**2 + (Y - vh//2)**2).astype(np.float32))
            max_dist = math.sqrt((vw/2)**2 + (vh/2)**2)
            if max_dist > 0:
                factor = np.clip(1.0 - (dist_map / max_dist) * 0.8, 0, 1)
                void_img_bgr = (void_img_bgr.astype(np.float32) * factor[:,:,np.newaxis]).astype(np.uint8)
            void_alpha = np.full((vh, vw), 255, dtype=np.uint8)
            void_img = np.dstack([void_img_bgr, void_alpha])
            canvas[void_y1:void_y1+vh, void_x1:void_x1+vw] = void_img

        def paste(bg, fg, x, y):
            fh, fw = fg.shape[:2]
            if x+fw > bg.shape[1] or y+fh > bg.shape[0]: return bg
            alpha = fg[:,:,3:4].astype(np.float32) / 255.0
            bg[y:y+fh, x:x+fw, :3] = (fg[:,:,:3].astype(np.float32) * alpha + bg[y:y+fh, x:x+fw, :3].astype(np.float32) * (1-alpha)).astype(np.uint8)
            bg[y:y+fh, x:x+fw, 3] = np.maximum(bg[y:y+fh, x:x+fw, 3], fg[:,:,3])
            return bg
        canvas = paste(canvas, sprite_a, pos_a_x, pos_a_y)
        canvas = paste(canvas, sprite_b, pos_b_x, pos_b_y)
        if target_rotation in (90, 270):
            canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        coords = cv2.findNonZero(canvas[:,:,3])
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            return canvas[y:y+h_crop, x:x+w_crop], (x, y, w_crop, h_crop)
        return canvas, (0,0,0,0)

    def render_profile_sprite(self, contours, size, rotation=0, profile_name="default"):
        if profile_name in SPECIAL_PAIRS_CONFIG:
            return self.create_paired_block(contours, size, profile_name, target_rotation=rotation)
        sprite, t = self.get_single_half(contours, size, rotation)
        if sprite is None: return np.zeros((10,10,4), dtype=np.uint8), (0,0,0,0)
        return sprite, (0,0,sprite.shape[1], sprite.shape[0])

# ==========================================
# PALET ÜRETİCİ (V14 - FAST)
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
            bg_path = random.choice(self.bg_images)
            bg = cv2.imread(bg_path)
            if bg is not None:
                bg = cv2.resize(bg, (self.width, self.height))
                return (bg.astype(np.float32) * random.uniform(0.5, 0.9)).astype(np.uint8)
        val = random.randint(40, 100)
        bg = np.full((self.height, self.width, 3), val, dtype=np.uint8)
        noise = np.random.randint(-30, 30, bg.shape, dtype=np.int16)
        return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def overlay_fast(self, img, overlay, pos_x, pos_y, alpha):
        h_c, w_c = img.shape[:2]
        oh, ow = overlay.shape[:2]
        px, py = int(pos_x), int(pos_y)
        x1, y1 = max(0, px), max(0, py)
        x2, y2 = min(w_c, px+ow), min(h_c, py+oh)
        if x2 <= x1 or y2 <= y1: return
        ox1, oy1 = x1-px, y1-py
        hs, ws = y2-y1, x2-x1
        a = alpha[oy1:oy1+hs, ox1:ox1+ws].astype(np.float32) / 255.0
        a3 = a[:,:,np.newaxis]
        roi = img[y1:y2, x1:x2]
        fg = overlay[oy1:oy1+hs, ox1:ox1+ws, :3]
        img[y1:y2, x1:x2] = (fg.astype(np.float32)*a3 + roi.astype(np.float32)*(1-a3)).astype(np.uint8)

    def check_collision(self, rect, rects):
        nx_r, ny_r, nw, nh = rect
        for ox, oy, ow, oh in rects:
            if nx_r < ox+ow and nx_r+nw > ox and ny_r < oy+oh and ny_r+nh > oy: return True
        return False

    def generate_pallet(self, ptype):
        pallet = self.get_random_background()
        anns = []
        contour = self.loader.profiles[ptype]
        sf = self.width / 640.0
        depth_cmds, face_cmds = [], []
        occupied = []

        for _ in range(random.randint(1, 2)):
            rot = 0
            bsz = int(random.randint(60, 70) * sf)
            if ptype == '1560':
                if random.random() < 0.5:
                    rot = 90; cols, rows = random.randint(1,2), random.randint(8,14)
                else:
                    cols, rows = random.randint(5,12), random.randint(1,4)
            else:
                if random.random() < 0.3: cols, rows = 1, random.randint(5,12)
                else: cols, rows = random.randint(3,6), random.randint(3,8)

            # CACHE: 1 kez üret
            cached, _ = self.renderer.render_profile_sprite(contour, bsz, rot, ptype)
            ph, pw = cached.shape[:2]
            if ph == 0 or pw == 0: continue
            pw_total, ph_total = cols*pw, rows*ph
            mx, my = max(0, self.width-pw_total), max(0, self.height-ph_total)
            found = False
            for _ in range(50):
                sx, sy = random.randint(0,mx), random.randint(0,my)
                r = [sx, sy, pw_total, ph_total]
                if not self.check_collision(r, occupied):
                    occupied.append(r); found = True; break
            if not found: continue

            cached_alpha = cached[:,:,3]
            for ri in range(rows):
                cy = sy + ri*ph
                for ci in range(cols):
                    cx = sx + ci*pw
                    if cx+pw > self.width or cy+ph > self.height: continue
                    fx, fy = cx+random.randint(-1,1), cy+random.randint(-1,1)
                    ocx, ocy = fx+pw//2, fy+ph//2
                    vx, vy = self.width//2-ocx, self.height//2-ocy
                    d = math.sqrt(vx**2+vy**2)
                    md = math.sqrt((self.width/2)**2+(self.height/2)**2)
                    dl = min((d/md)*(50*sf), pw*0.8)
                    depth_cmds.append({'x':fx,'y':fy,'sw':pw,'sh':ph,'vx':vx,'vy':vy,'d':d,'dl':dl,'alpha':cached_alpha})
                    face_cmds.append({'sprite':cached,'x':fx,'y':fy,'sw':pw,'sh':ph})

        # DEPTH (reduced steps, step=2)
        for c in depth_cmds:
            if c['d'] <= 0: continue
            nx_v, ny_v = c['vx']/c['d'], c['vy']/c['d']
            tox, toy = int(nx_v*c['dl']), int(ny_v*c['dl'])
            body = np.full((c['sh'], c['sw'], 3), 50, dtype=np.uint8)
            steps = min(int(max(abs(tox), abs(toy))), 15)
            for s in range(steps, 0, -2):
                r = s/steps if steps > 0 else 0
                self.overlay_fast(pallet, body, c['x']+int(tox*r), c['y']+int(toy*r), c['alpha'])

        cls_id = CLASS_MAP[ptype]
        for c in face_cmds:
            self.overlay_fast(pallet, c['sprite'], c['x'], c['y'], c['sprite'][:,:,3])
            anns.append({'class_id': cls_id, 'bbox': [
                (c['x']+c['sw']/2)/self.width, (c['y']+c['sh']/2)/self.height,
                c['sw']/self.width, c['sh']/self.height]})
        return pallet, anns

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def main():
    loader = DXFProfileLoader()
    for name, path in MY_PROFILES.items(): loader.load_dxf(path, name)
    if not loader.profiles: print("❌ Profil yüklenemedi!"); return
    renderer = ProfileRenderer()
    gen = SyntheticPalletGenerator(loader, renderer, BG_FOLDER)
    for s in ['train','val']:
        os.makedirs(f'{BASE_DIR}/images/{s}', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/labels/{s}', exist_ok=True)
    print(f"\n🚀 Script 1/3 FAST: 1524 + 1560 Başlıyor...")
    for ptype in MY_PROFILES:
        if ptype not in loader.profiles: continue
        print(f"\n📦 {ptype} ...")
        for i in range(IMAGES_PER_CLASS_TRAIN):
            if i % 20 == 0: print(f"   [Train] {i}/{IMAGES_PER_CLASS_TRAIN}")
            p, a = gen.generate_pallet(ptype)
            n = f"{ptype}_train_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/train/{n}.jpg', p)
            #cv2.imwrite(f'{BASE_DIR}/images/train/{n}_clahe.jpg', apply_clahe(p))
            ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
            #for suffix in [f'{n}.txt', f'{n}_clahe.txt']:
            for suffix in [f'{n}.txt']:
                with open(f'{BASE_DIR}/labels/train/{suffix}', 'w') as f: f.write(ls)
        for i in range(IMAGES_PER_CLASS_VAL):
            if i % 5 == 0: print(f"   [Val] {i}/{IMAGES_PER_CLASS_VAL}")
            p, a = gen.generate_pallet(ptype)
            n = f"{ptype}_val_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/val/{n}.jpg', p)
            ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
            with open(f'{BASE_DIR}/labels/val/{n}.txt', 'w') as f: f.write(ls)
    print(f"\n✅ Script 1/3 TAMAMLANDI!")

if __name__ == "__main__":
    main()
