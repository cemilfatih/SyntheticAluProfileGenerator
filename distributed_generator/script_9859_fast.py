"""
Script 2/3: 9859 Profili (TURBO)
Kaynak: Test Tuner - Package + Pattern + Tunnel Void

HIZLANDIRMA:
1. DXF parse + mask + texture = 1 KEZ (başta)
2. Paket sprite = 1 KEZ (başta) 
3. Flipped paket = 1 KEZ (başta)
4. Tunnel void = boyut bazlı cache (aynı boyut tekrar hesaplanmaz)
5. Canvas boyutu kontrollü (gereksiz büyük canvas yok)
6. Depth step azaltıldı + step=2
7. Texture vectorized
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
IMAGES_PER_CLASS_TRAIN = 100
IMAGES_PER_CLASS_VAL = 20
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
# DXF -> MASK (1 kez çalışır)
# ==========================================
def extract_mask_from_dxf(dxf_path, canvas_size):
    """DXF dosyasını okur, XOR polygon mask döndürür."""
    if not os.path.exists(dxf_path):
        return None
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    lines = []
    for e in msp:
        if e.dxftype() == 'LINE':
            lines.append(((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)))
        elif e.dxftype() == 'LWPOLYLINE':
            pts = e.get_points('xy')
            for i in range(len(pts)-1):
                lines.append((pts[i], pts[i+1]))
            if e.closed:
                lines.append((pts[-1], pts[0]))
        elif e.dxftype() == 'ARC':
            c, r = e.dxf.center, e.dxf.radius
            sa, ea = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
            if ea < sa: ea += 2 * math.pi
            pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(sa, ea, 30)]
            for i in range(len(pts)-1):
                lines.append((pts[i], pts[i+1]))
        elif e.dxftype() == 'CIRCLE':
            c, r = e.dxf.center, e.dxf.radius
            pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 60)]
            for i in range(len(pts)-1):
                lines.append((pts[i], pts[i+1]))
    if not lines:
        return None

    all_pts = np.array([pt for p1, p2 in lines for pt in (p1, p2)])
    min_vals, max_vals = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
    max_dim = max(max_vals[0]-min_vals[0], max_vals[1]-min_vals[1])
    if max_dim == 0: max_dim = 1
    padding = int(canvas_size * 0.1)
    scale = (canvas_size - 2*padding) / max_dim

    scaled_lines = []
    for p1, p2 in lines:
        x1 = (p1[0]-min_vals[0])*scale + padding
        y1 = canvas_size - ((p1[1]-min_vals[1])*scale + padding)
        x2 = (p2[0]-min_vals[0])*scale + padding
        y2 = canvas_size - ((p2[1]-min_vals[1])*scale + padding)
        scaled_lines.append(((x1, y1), (x2, y2)))

    # Edge chaining -> polygon
    polygons = get_polygons_from_lines(scaled_lines, tolerance=3.0)
    polygons = sorted(polygons, key=cv2.contourArea, reverse=True)
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    for poly in polygons:
        temp = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(temp, [poly], 255)
        mask = cv2.bitwise_xor(mask, temp)
    return mask


def get_polygons_from_lines(lines, tolerance=3.0):
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
                if d1 < best_dist:
                    best_dist, best_idx, best_reverse = d1, i, False
                if d2 < best_dist:
                    best_dist, best_idx, best_reverse = d2, i, True
            if best_idx != -1 and best_dist <= tolerance:
                edge = edges.pop(best_idx)
                current_poly.append(edge[0] if best_reverse else edge[1])
            else:
                break
        if len(current_poly) > 2:
            polygons.append(np.array(current_poly, dtype=np.int32))
    return polygons


def apply_aluminum_texture(mask):
    """Vectorized alüminyum dokusu."""
    h, w = mask.shape
    is_shiny = random.random() < 0.3
    base_val = random.randint(200, 240) if is_shiny else random.randint(150, 190)
    contrast = 30 if is_shiny else 15
    if random.random() < 0.5:
        grad = np.clip(base_val - contrast * np.sin(np.arange(h, dtype=np.float32) / h * 6.28), 0, 255).astype(np.uint8)
        img_bgr = np.tile(grad[:, np.newaxis, np.newaxis], (1, w, 3))
    else:
        grad = np.clip(base_val - contrast * np.sin(np.arange(w, dtype=np.float32) / w * 6.28), 0, 255).astype(np.uint8)
        img_bgr = np.tile(grad[np.newaxis, :, np.newaxis], (h, 1, 3))
    noise = np.random.randint(-12, 12, img_bgr.shape, dtype=np.int16)
    img_bgr = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_bgr, contours_cv, -1, (50, 50, 50), 1, cv2.LINE_AA)
    return cv2.merge((img_bgr, mask))


def build_base_sprite(dxf_path, size):
    """DXF'den tek profil sprite'ı üretir. 1 KEZ çağrılır."""
    canvas_size = int(size * 3.0)
    mask = extract_mask_from_dxf(dxf_path, canvas_size)
    if mask is None:
        return None
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    textured = apply_aluminum_texture(mask[y:y+h, x:x+w])
    # Yataysa dik çevir
    if textured.shape[1] > textured.shape[0]:
        textured = cv2.rotate(textured, cv2.ROTATE_90_CLOCKWISE)
    return textured


def build_package(sprite, n=6):
    """6 profili yan yana birleştirip paket yapar. 1 KEZ çağrılır."""
    h, w = sprite.shape[:2]
    pkg = np.zeros((h, w * n, 4), dtype=np.uint8)
    for i in range(n):
        pkg[:, i*w:(i+1)*w] = sprite
    return pkg


# ==========================================
# PALET ÜRETİCİ
# ==========================================
class PalletGenerator:
    def __init__(self, pkg_normal, pkg_flipped, pkg_rotated, bg_folder):
        """
        Önceden üretilmiş 3 paket varyantını alır:
        - pkg_normal: düz paket
        - pkg_flipped: 180° döndürülmüş (alternating satırlar için)
        - pkg_rotated: 90° döndürülmüş (vertical satırlar için)
        """
        self.pkg_normal = pkg_normal
        self.pkg_flipped = pkg_flipped
        self.pkg_rotated = pkg_rotated
        self.pkg_h, self.pkg_w = pkg_normal.shape[:2]
        self.rot_h, self.rot_w = pkg_rotated.shape[:2]
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.bg_images = glob.glob(os.path.join(bg_folder, "*.*"))
        # Tunnel void cache: (w,h) -> void template
        self._void_cache = {}

    def overlay_fast(self, img, overlay, px, py, alpha):
        hc, wc = img.shape[:2]
        oh, ow = overlay.shape[:2]
        px, py = int(px), int(py)
        x1, y1 = max(0, px), max(0, py)
        x2, y2 = min(wc, px+ow), min(hc, py+oh)
        if x2 <= x1 or y2 <= y1:
            return
        ox, oy = x1-px, y1-py
        hs, ws = y2-y1, x2-x1
        a = alpha[oy:oy+hs, ox:ox+ws].astype(np.float32) / 255.0
        a3 = a[:, :, np.newaxis]
        fg = overlay[oy:oy+hs, ox:ox+ws, :3].astype(np.float32)
        bg = img[y1:y2, x1:x2].astype(np.float32)
        img[y1:y2, x1:x2] = (fg*a3 + bg*(1-a3)).astype(np.uint8)

    def get_tunnel_void(self, w, h, ocx, ocy, cw, ch):
        """Boyut bazlı cached tunnel void."""
        vw, vh = w-4, h-4
        if vw <= 0 or vh <= 0:
            return None
        
        # Işık yönü her seferinde farklı olsun diye cache kullanmayalım
        # ama en azından np hesaplarını basitleştirelim
        vx, vy = cw//2 - ocx, ch//2 - ocy
        lcx = np.clip(vw//2 + int(vx*0.3), 0, vw)
        lcy = np.clip(vh//2 + int(vy*0.3), 0, vh)
        
        # Boyut bazlı distance map cache
        key = (vw, vh)
        if key not in self._void_cache:
            Y, X = np.ogrid[:vh, :vw]
            # center (vw//2, vh//2) için base distance map
            base_dm = np.sqrt(((X - vw//2)**2 + (Y - vh//2)**2).astype(np.float32))
            md = np.sqrt(float(vw**2 + vh**2))
            self._void_cache[key] = (base_dm, md, Y, X)
        
        _, md, Y, X = self._void_cache[key]
        # Işık merkezine göre distance map (her seferinde farklı)
        dm = np.sqrt(((X - lcx)**2 + (Y - lcy)**2).astype(np.float32))
        nd = np.clip(dm / (md * 0.6), 0, 1)
        alpha = (np.power(nd, 0.7) * random.uniform(0.80, 0.98) * 255).astype(np.uint8)
        base_color = np.zeros((vh, vw, 3), dtype=np.uint8)
        return np.dstack([base_color, alpha])

    def get_random_background(self, w, h):
        if self.bg_images and random.random() < 0.9:
            bg = cv2.imread(random.choice(self.bg_images))
            if bg is not None:
                bg = cv2.resize(bg, (w, h))
                return (bg.astype(np.float32) * random.uniform(0.5, 0.95)).astype(np.uint8)
        val = random.randint(50, 110)
        bg = np.full((h, w, 3), val, dtype=np.uint8)
        noise = np.random.randint(-25, 25, bg.shape, dtype=np.int16)
        return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def generate_pallet(self):
        cls_id = CLASS_MAP['9859']
        
        pattern = random.choices(["ALTERNATING", "SANDWICH", "UNIFORM"],
                                 weights=[0.70, 0.20, 0.10], k=1)[0]
        num_rows = random.randint(8, 18)
        pkgs_per_row = random.randint(3, 6)
        pile_width = pkgs_per_row * self.pkg_w

        # Canvas boyutunu makul tut
        canvas_w = pile_width + random.randint(200, 400)
        canvas_h = (num_rows + 1) * max(self.pkg_h, self.rot_h) + 200
        # Minimum 1280
        canvas_w = max(canvas_w, self.width)
        canvas_h = max(canvas_h, self.height)

        wall = self.get_random_background(canvas_w, canvas_h)
        cur_y = canvas_h - random.randint(80, 150)
        start_x = (canvas_w - pile_width) // 2

        depth_cmds = []
        face_cmds = []

        for row_idx in range(num_rows):
            rtype = "HORIZONTAL"
            flip = False
            if pattern == "ALTERNATING" and row_idx % 2 == 1:
                flip = True
            elif pattern == "SANDWICH" and (num_rows//3 <= row_idx < 2*(num_rows//3)):
                rtype = "VERTICAL"

            if rtype == "HORIZONTAL":
                cur_y -= self.pkg_h
                if cur_y < 0:
                    break
                stamp = self.pkg_flipped if flip else self.pkg_normal
                stamp_alpha = stamp[:, :, 3]
                sh, sw = stamp.shape[:2]
                for col in range(pkgs_per_row):
                    x = start_x + col * self.pkg_w
                    self._add_commands(depth_cmds, face_cmds, stamp, stamp_alpha,
                                       x, cur_y, sw, sh, canvas_w, canvas_h)
            else:
                stamp = self.pkg_rotated
                stamp_alpha = stamp[:, :, 3]
                rh, rw = stamp.shape[:2]
                cur_y -= rh
                if cur_y < 0:
                    break
                v_pkgs = pile_width // rw if rw > 0 else 1
                off_x = start_x + (pile_width - v_pkgs * rw) // 2
                for col in range(v_pkgs):
                    x = off_x + col * rw
                    self._add_commands(depth_cmds, face_cmds, stamp, stamp_alpha,
                                       x, cur_y, rw, rh, canvas_w, canvas_h)

        # RENDER: Depth (reduced steps)
        for c in depth_cmds:
            if c['d'] <= 0:
                continue
            nx_v, ny_v = c['vx']/c['d'], c['vy']/c['d']
            tox = int(nx_v * c['dl'])
            toy = int(ny_v * c['dl'])
            body = np.full((c['sh'], c['sw'], 3), random.randint(30, 60), dtype=np.uint8)
            steps = min(int(max(abs(tox), abs(toy))), 12)
            for s in range(steps, 0, -2):
                r = s / steps if steps > 0 else 0
                self.overlay_fast(wall, body, c['x']+int(tox*r), c['y']+int(toy*r), c['alpha'])
            # Void
            if c['void'] is not None:
                self.overlay_fast(wall, c['void'], c['x']+2, c['y']+2, c['void'][:,:,3])

        # RENDER: Face
        for c in face_cmds:
            self.overlay_fast(wall, c['sprite'], c['x'], c['y'], c['sprite'][:,:,3])

        # Crop & resize to 1280x1280
        crop_top = max(0, cur_y - 100)
        cropped = wall[crop_top:canvas_h, :]
        final = cv2.resize(cropped, (self.width, self.height))

        # Annotations
        anns = []
        scale_x = self.width / cropped.shape[1]
        scale_y = self.height / cropped.shape[0]

        for c in face_cmds:
            sw_c, sh_c = c['sw'], c['sh']
            px_c = c['x']
            py_c = c['y'] - crop_top  # crop offset

            for i in range(PROFILES_PER_PKG):
                if sw_c > sh_c:
                    # Yatay paket
                    bw = sw_c / float(PROFILES_PER_PKG)
                    bh = float(sh_c)
                    bx = px_c + i * bw
                    by = float(py_c)
                else:
                    # Dikey paket (rotated)
                    bw = float(sw_c)
                    bh = sh_c / float(PROFILES_PER_PKG)
                    bx = float(px_c)
                    by = py_c + i * bh

                cx_n = ((bx + bw/2) * scale_x) / self.width
                cy_n = ((by + bh/2) * scale_y) / self.height
                nw = (bw * scale_x) / self.width
                nh = (bh * scale_y) / self.height

                if 0 <= cx_n <= 1 and 0 <= cy_n <= 1 and nw > 0 and nh > 0:
                    anns.append({'class_id': cls_id,
                                 'bbox': [cx_n, cy_n, nw, nh]})

        return final, anns

    def _add_commands(self, dl, fl, stamp, stamp_alpha, x, y, sw, sh, cw, ch):
        ocx, ocy = x + sw//2, y + sh//2
        vx, vy = cw//2 - ocx, ch//2 - ocy
        d = math.sqrt(vx**2 + vy**2)
        md = math.sqrt((cw/2)**2 + (ch/2)**2)
        dlen = min((d/md) * 60, sw * 0.8)
        void = self.get_tunnel_void(sw, sh, ocx, ocy, cw, ch)
        dl.append({
            'x': x, 'y': y, 'sw': sw, 'sh': sh,
            'alpha': stamp_alpha,
            'vx': vx, 'vy': vy, 'd': d, 'dl': dlen,
            'void': void
        })
        fl.append({
            'sprite': stamp, 'x': x, 'y': y,
            'sw': sw, 'sh': sh
        })


# ==========================================
# MAIN
# ==========================================
def main():
    if not os.path.exists(DXF_PROFILE):
        print(f"❌ {DXF_PROFILE} bulunamadı!")
        return

    print(f"\n🔧 9859 sprite hazırlanıyor (1 kez)...")
    
    # === 1 KEZ: DXF -> Sprite -> Paket ===
    base_sprite = build_base_sprite(DXF_PROFILE, RENDER_SIZE)
    if base_sprite is None:
        print("❌ Sprite üretilemedi!")
        return
    
    pkg_normal = build_package(base_sprite, PROFILES_PER_PKG)
    pkg_flipped = cv2.rotate(pkg_normal, cv2.ROTATE_180)
    pkg_rotated = cv2.rotate(pkg_normal, cv2.ROTATE_90_CLOCKWISE)
    
    print(f"   Sprite: {base_sprite.shape[1]}x{base_sprite.shape[0]}")
    print(f"   Paket:  {pkg_normal.shape[1]}x{pkg_normal.shape[0]}")
    print(f"   ✅ Hazır!")

    gen = PalletGenerator(pkg_normal, pkg_flipped, pkg_rotated, BG_FOLDER)

    for s in ['train', 'val']:
        os.makedirs(f'{BASE_DIR}/images/{s}', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/labels/{s}', exist_ok=True)

    print(f"\n🚀 Script 2/3 TURBO: 9859 Başlıyor...")

    for i in range(IMAGES_PER_CLASS_TRAIN):
        if i % 20 == 0:
            print(f"   [Train] {i}/{IMAGES_PER_CLASS_TRAIN}")
        p, a = gen.generate_pallet()
        n = f"9859_train_{i}"
        cv2.imwrite(f'{BASE_DIR}/images/train/{n}.jpg', p)
        ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
        with open(f'{BASE_DIR}/labels/train/{n}.txt', 'w') as f:
            f.write(ls)

    for i in range(IMAGES_PER_CLASS_VAL):
        if i % 5 == 0:
            print(f"   [Val] {i}/{IMAGES_PER_CLASS_VAL}")
        p, a = gen.generate_pallet()
        n = f"9859_val_{i}"
        cv2.imwrite(f'{BASE_DIR}/images/val/{n}.jpg', p)
        ls = "".join([f"{x['class_id']} {x['bbox'][0]:.6f} {x['bbox'][1]:.6f} {x['bbox'][2]:.6f} {x['bbox'][3]:.6f}\n" for x in a])
        with open(f'{BASE_DIR}/labels/val/{n}.txt', 'w') as f:
            f.write(ls)

    print(f"\n✅ Script 2/3 TAMAMLANDI!")


if __name__ == "__main__":
    main()