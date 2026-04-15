import ezdxf
import cv2
import numpy as np
import random
import os
import math
import glob
import shutil
import networkx as nx

# ==========================================
# --- 1. AYARLAR VE STATİK DOSYA YOLLARI ---
# ==========================================
IMAGES_PER_CLASS_TRAIN = 10
IMAGES_PER_CLASS_VAL = 1
IMG_WIDTH = 1280
IMG_HEIGHT = 1280

BASE_DIR = "datasets/claude_dataset_5_profile_FINAL"
BG_FOLDER = "background"
DXF_FOLDER = "cad_files"

# SADECE BU 5 DOSYAYA BAKILACAK
STATIC_PROFILES = {
    '1524': os.path.join(DXF_FOLDER, "1524.dxf"),
    '1560': os.path.join(DXF_FOLDER, "1560.dxf"),
    '9859': os.path.join(DXF_FOLDER, "9859.dxf"),
    '7170': os.path.join(DXF_FOLDER, "7170.dxf"),
    '9794': os.path.join(DXF_FOLDER, "9794.dxf"),
}

# İkili eşleşen profiller (Script 1 - V14)
SPECIAL_PAIRS_CONFIG = {
    '1524': 0.70,
    '1560': 0.85,
}

# Paketli profiller (Script 2 - 9859 Test)
PACKAGE_PROFILES = {'9859': 6}

# Sınıf haritası (sabit sıra)
CLASS_MAP = {name: i for i, name in enumerate(STATIC_PROFILES.keys())}


# ==========================================
# --- 2. DXF YÜKLEYİCİ ---
# ==========================================
class DXFProfileLoader:
    """
    İki farklı yükleme stratejisi:
    - 1524/1560: Multi-contour graph tabanlı (V14 - iç içe geçen parçalar için)
    - 7170/9794: Single-contour graph tabanlı (V24 - basit profiller için)
    - 9859: Line tabanlı + XOR polygon (Test Tuner - delikli profil için)
    """

    def __init__(self):
        self.profiles_multi = {}   # 1524, 1560 -> list of contour arrays
        self.profiles_single = {}  # 7170, 9794 -> single contour array
        self.lines_raw = {}        # 9859 -> raw normalized lines

    def load_all(self):
        for name, path in STATIC_PROFILES.items():
            if not os.path.exists(path):
                print(f"⚠️  {path} bulunamadı, atlanıyor.")
                continue
            try:
                if name in SPECIAL_PAIRS_CONFIG:
                    self._load_multi_contour(path, name)
                elif name in PACKAGE_PROFILES:
                    self._load_line_based(path, name)
                else:
                    self._load_single_contour(path, name)
            except Exception as e:
                print(f"❌ DXF Hatası ({name}): {e}")

    # --- 1524/1560: V14'teki multi-contour graph mantığı ---
    def _load_multi_contour(self, dxf_path, profile_name):
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        G = nx.Graph()

        def to_key(p): return (round(p[0], 3), round(p[1], 3))

        for e in msp:
            if e.dxftype() == 'LINE':
                G.add_edge(to_key(e.dxf.start), to_key(e.dxf.end))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points('xy')
                for i in range(len(pts) - 1):
                    G.add_edge(to_key(pts[i]), to_key(pts[i + 1]))
                if e.closed:
                    G.add_edge(to_key(pts[-1]), to_key(pts[0]))
            elif e.dxftype() == 'ARC':
                c, r = e.dxf.center, e.dxf.radius
                sa = math.radians(e.dxf.start_angle)
                ea = math.radians(e.dxf.end_angle)
                if ea < sa: ea += 2 * math.pi
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)]
                       for a in np.linspace(sa, ea, 30)]
                for i in range(len(pts) - 1):
                    G.add_edge(to_key(pts[i]), to_key(pts[i + 1]))
            elif e.dxftype() == 'CIRCLE':
                c, r = e.dxf.center, e.dxf.radius
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)]
                       for a in np.linspace(0, 2 * math.pi, 60)]
                for i in range(len(pts) - 1):
                    G.add_edge(to_key(pts[i]), to_key(pts[i + 1]))
                G.add_edge(to_key(pts[-1]), to_key(pts[0]))

        if len(G.nodes) == 0:
            return

        components = list(nx.connected_components(G))
        main_comp = max(components, key=len)
        main_pts = np.array(list(main_comp))
        main_min = np.min(main_pts, axis=0)
        main_max = np.max(main_pts, axis=0)

        valid_comps = []
        for comp in components:
            if len(comp) < 3:
                continue
            pts = np.array(list(comp))
            c_min, c_max = np.min(pts, axis=0), np.max(pts, axis=0)
            if comp == main_comp or (np.all(c_min >= main_min - 0.1) and np.all(c_max <= main_max + 0.1)):
                valid_comps.append(comp)

        if not valid_comps:
            return

        all_pts_list = []
        contours = []
        for comp in valid_comps:
            subgraph = G.subgraph(comp)
            start_node = next((n for n, d in subgraph.degree() if d == 1), list(comp)[0])
            ordered = list(nx.dfs_preorder_nodes(subgraph, source=start_node))
            contours.append(np.array(ordered))
            all_pts_list.extend(ordered)

        all_pts = np.array(all_pts_list)
        min_vals = np.min(all_pts, axis=0)
        max_vals = np.max(all_pts, axis=0)
        center = (min_vals + max_vals) / 2
        max_dim = np.max(max_vals - min_vals)

        normalized = []
        for pts in contours:
            pts = pts - center
            if max_dim > 0:
                pts = pts / max_dim
            normalized.append(pts)

        self.profiles_multi[profile_name] = normalized
        print(f"✅ Yüklendi: {profile_name} (multi-contour, {len(normalized)} parça)")

    # --- 7170/9794: V24'teki single-contour graph mantığı ---
    def _load_single_contour(self, dxf_path, profile_name):
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        G = nx.Graph()

        def to_key(p): return (round(p[0], 3), round(p[1], 3))

        for e in msp:
            if e.dxftype() == 'LINE':
                G.add_edge(to_key(e.dxf.start), to_key(e.dxf.end))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points('xy')
                for i in range(len(pts) - 1):
                    G.add_edge(to_key(pts[i]), to_key(pts[i + 1]))
                if e.closed:
                    G.add_edge(to_key(pts[-1]), to_key(pts[0]))

        if len(G.nodes) == 0:
            return

        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        subgraph = G.subgraph(largest)
        ordered = list(nx.dfs_preorder_nodes(subgraph))

        pts = np.array(ordered)
        min_vals = np.min(pts, axis=0)
        max_vals = np.max(pts, axis=0)
        center = (min_vals + max_vals) / 2
        pts = pts - center
        h = max_vals[1] - min_vals[1]
        if h > 0:
            pts = pts / h

        self.profiles_single[profile_name] = pts
        print(f"✅ Yüklendi: {profile_name} (single-contour)")

    # --- 9859: Test Tuner'daki line tabanlı + XOR polygon mantığı ---
    def _load_line_based(self, dxf_path, profile_name):
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        lines = []

        for e in msp:
            if e.dxftype() == 'LINE':
                lines.append(((e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points('xy')
                for i in range(len(pts) - 1):
                    lines.append((pts[i], pts[i + 1]))
                if e.closed:
                    lines.append((pts[-1], pts[0]))
            elif e.dxftype() == 'ARC':
                c, r = e.dxf.center, e.dxf.radius
                sa = math.radians(e.dxf.start_angle)
                ea = math.radians(e.dxf.end_angle)
                if ea < sa: ea += 2 * math.pi
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)]
                       for a in np.linspace(sa, ea, 30)]
                for i in range(len(pts) - 1):
                    lines.append((pts[i], pts[i + 1]))
            elif e.dxftype() == 'CIRCLE':
                c, r = e.dxf.center, e.dxf.radius
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)]
                       for a in np.linspace(0, 2 * math.pi, 60)]
                for i in range(len(pts) - 1):
                    lines.append((pts[i], pts[i + 1]))

        if not lines:
            return

        self.lines_raw[profile_name] = lines
        print(f"✅ Yüklendi: {profile_name} (line-based, {len(lines)} çizgi)")

    def has_profile(self, name):
        return (name in self.profiles_multi or
                name in self.profiles_single or
                name in self.lines_raw)


# ==========================================
# --- 3. RENDER MOTORU ---
# ==========================================
class ProfileRenderer:
    def __init__(self, loader):
        self.loader = loader

    # --- ORTAK: Alüminyum dokusu ---
    def apply_aluminum_texture(self, mask, profile_name=""):
        h, w = mask.shape
        if profile_name == '9859':
            is_shiny = random.random() < 0.3
            base_val = random.randint(200, 240) if is_shiny else random.randint(150, 190)
            contrast = 30 if is_shiny else 15
        else:
            is_shiny = random.random() < 0.30
            base_val = random.randint(230, 255) if is_shiny else random.randint(180, 220)
            contrast = 40 if is_shiny else 25

        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        if random.random() < 0.5:
            for y in range(h):
                v = int(np.clip(base_val - contrast * np.sin(y / max(h, 1) * 3.14), 0, 255))
                img_bgr[y, :, :] = (v, v, v)
        else:
            for x in range(w):
                v = int(np.clip(base_val - contrast * np.sin(x / max(w, 1) * 3.14), 0, 255))
                img_bgr[:, x, :] = (v, v, v)

        noise = np.random.randint(-15, 15, img_bgr.shape)
        img_bgr = np.clip(img_bgr.astype(int) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

        contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours_cv, -1, (50, 50, 50), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    # ========================================
    # 1524/1560: Multi-contour paired block
    # (V14'teki get_single_half + create_paired_block)
    # ========================================
    def get_single_half_multi(self, contours_list, size, rotation=0):
        """V14: Birden fazla contour'u tek mask'e çizer."""
        canvas_size = int(size * 2.5)
        pts_list = []
        for contour in contours_list:
            scaled = contour * size + canvas_size // 2
            if rotation != 0:
                M = cv2.getRotationMatrix2D((canvas_size // 2, canvas_size // 2), rotation, 1.0)
                ones = np.ones((len(scaled), 1))
                scaled = M.dot(np.hstack([scaled, ones]).T).T
            pts_list.append(scaled.astype(np.int32))

        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(mask, pts_list, 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None, 0
        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y + h, x:x + w]
        thickness = int(w * 0.12)
        return self.apply_aluminum_texture(cropped_mask), thickness

    def create_paired_block(self, contours_list, size, profile_name, target_rotation=0):
        """V14: İkili eşleşme bloğu (1524/1560)."""
        sprite_a, t = self.get_single_half_multi(contours_list, size, 0)
        if sprite_a is None:
            return np.zeros((10, 10, 4), dtype=np.uint8), 0
        h, w = sprite_a.shape[:2]

        sprite_b_raw, _ = self.get_single_half_multi(contours_list, size, 180)
        if sprite_b_raw is None:
            return sprite_a, t
        sprite_b = cv2.resize(sprite_b_raw, (w, h))

        tightness = SPECIAL_PAIRS_CONFIG.get(profile_name, 1.0)
        shift_val = int(t * tightness)

        canvas_w = w * 2 + shift_val * 4
        canvas_h = h * 2 + shift_val * 4
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        cx, cy = canvas_w // 2, canvas_h // 2

        pos_a_x, pos_a_y = cx - w // 2, cy - h // 2
        pos_b_x, pos_b_y = pos_a_x + shift_val, pos_a_y + shift_val

        # --- VOID FILLING (V14'teki noisy greyscale) ---
        void_x1 = min(pos_a_x, pos_b_x) + t
        void_y1 = min(pos_a_y, pos_b_y) + t
        void_x2 = max(pos_a_x + w, pos_b_x + w) - t
        void_y2 = max(pos_a_y + h, pos_b_y + h) - t
        if void_x2 > void_x1 and void_y2 > void_y1:
            vw = void_x2 - void_x1
            vh = void_y2 - void_y1
            if profile_name == '1560':
                base_val_arr = np.random.randint(50, 90, (vh, vw), dtype=np.uint8)
            else:
                base_val_arr = np.random.randint(10, 30, (vh, vw), dtype=np.uint8)
            void_img_bgr = cv2.merge([base_val_arr, base_val_arr, base_val_arr])
            noise = np.random.randint(-20, 20, (vh, vw, 3))
            void_img_bgr = np.clip(void_img_bgr.astype(int) + noise, 0, 255).astype(np.uint8)

            # Merkeze doğru kararma
            center_v = (vw // 2, vh // 2)
            max_dist = math.sqrt((vw / 2) ** 2 + (vh / 2) ** 2)
            if max_dist > 0:
                Y, X = np.ogrid[:vh, :vw]
                dist_map = np.sqrt((X - center_v[0]) ** 2 + (Y - center_v[1]) ** 2)
                factor_map = 1.0 - (dist_map / max_dist) * 0.8
                factor_map = np.clip(factor_map, 0, 1)
                for c_ch in range(3):
                    void_img_bgr[:, :, c_ch] = (void_img_bgr[:, :, c_ch].astype(float) * factor_map).astype(np.uint8)

            void_alpha = np.full((vh, vw), 255, dtype=np.uint8)
            void_img = cv2.merge([void_img_bgr[:, :, 0], void_img_bgr[:, :, 1],
                                  void_img_bgr[:, :, 2], void_alpha])
            canvas[void_y1:void_y1 + vh, void_x1:void_x1 + vw] = void_img

        def paste(bg, fg, x, y):
            fh, fw = fg.shape[:2]
            if x + fw > bg.shape[1] or y + fh > bg.shape[0]:
                return bg
            alpha = fg[:, :, 3] / 255.0
            for c_ch in range(3):
                bg[y:y + fh, x:x + fw, c_ch] = (
                    fg[:, :, c_ch] * alpha + bg[y:y + fh, x:x + fw, c_ch] * (1 - alpha)
                )
            bg[y:y + fh, x:x + fw, 3] = np.maximum(bg[y:y + fh, x:x + fw, 3], fg[:, :, 3])
            return bg

        canvas = paste(canvas, sprite_a, pos_a_x, pos_a_y)
        canvas = paste(canvas, sprite_b, pos_b_x, pos_b_y)

        if target_rotation == 90 or target_rotation == 270:
            canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

        coords = cv2.findNonZero(canvas[:, :, 3])
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            return canvas[y:y + h_crop, x:x + w_crop], t
        return canvas, 0

    # ========================================
    # 7170/9794: Single-contour render
    # (V24'teki get_single_half)
    # ========================================
    def get_single_half_single(self, contour, size, rotation=0, profile_name=""):
        """V24: Tek contour'dan sprite üretir."""
        canvas_size = int(size * 2.5)
        scaled = contour * size + canvas_size // 2
        if rotation != 0:
            M = cv2.getRotationMatrix2D((canvas_size // 2, canvas_size // 2), rotation, 1.0)
            ones = np.ones((len(scaled), 1))
            scaled = M.dot(np.hstack([scaled, ones]).T).T
        pts = scaled.astype(np.int32)
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None, 0
        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y + h, x:x + w]
        thickness = int(min(w, h) * 0.15)
        return self.apply_aluminum_texture(cropped_mask, profile_name), thickness

    # ========================================
    # 9859: Test Tuner'daki XOR polygon mask
    # ========================================
    def get_9859_sprite(self, size):
        """Test Tuner: 9859 için XOR polygon tabanlı mask."""
        lines = self.loader.lines_raw.get('9859')
        if not lines:
            return None

        canvas_size = int(size * 3.0)
        all_pts = np.array([pt for p1, p2 in lines for pt in (p1, p2)])
        min_vals = np.min(all_pts, axis=0)
        max_vals = np.max(all_pts, axis=0)
        max_dim = max(max_vals[0] - min_vals[0], max_vals[1] - min_vals[1])
        if max_dim == 0:
            max_dim = 1

        padding = int(canvas_size * 0.1)
        scale = (canvas_size - 2 * padding) / max_dim

        scaled_lines = []
        for p1, p2 in lines:
            x1 = (p1[0] - min_vals[0]) * scale + padding
            y1 = canvas_size - ((p1[1] - min_vals[1]) * scale + padding)
            x2 = (p2[0] - min_vals[0]) * scale + padding
            y2 = canvas_size - ((p2[1] - min_vals[1]) * scale + padding)
            scaled_lines.append((np.array([x1, y1]), np.array([x2, y2])))

        # Edge chaining -> polygon
        polygons = self._get_polygons_from_lines(scaled_lines, tolerance=3.0)
        polygons = sorted(polygons, key=cv2.contourArea, reverse=True)

        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        for poly in polygons:
            temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [poly], 255)
            mask = cv2.bitwise_xor(mask, temp_mask)

        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y + h, x:x + w]
        textured = self.apply_aluminum_texture(cropped_mask, '9859')

        # Yataysa dik çevir
        if textured.shape[1] > textured.shape[0]:
            textured = cv2.rotate(textured, cv2.ROTATE_90_CLOCKWISE)
        return textured

    def _get_polygons_from_lines(self, lines, tolerance=3.0):
        """Test Tuner: Kenarları takip ederek poligon oluşturur."""
        edges = [[np.array(p1, dtype=np.float64), np.array(p2, dtype=np.float64)]
                 for p1, p2 in lines]
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

    def create_package_9859(self, sprite, count=6):
        """Test Tuner: 6 profili yan yana birleştirip paket yapar."""
        h, w = sprite.shape[:2]
        pkg = np.zeros((h, w * count, 4), dtype=np.uint8)
        for i in range(count):
            pkg[:, i * w:(i + 1) * w] = sprite
        return pkg


# ==========================================
# --- 4. PALET ÜRETİCİ ---
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
                factor = random.uniform(0.5, 0.9)
                return (bg.astype(float) * factor).astype(np.uint8)
        val = random.randint(40, 100)
        bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * val
        noise = np.random.randint(-30, 30, bg.shape)
        return np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)

    def overlay_image_alpha(self, img, img_overlay, pos_x, pos_y, alpha_mask):
        canvas_h, canvas_w = img.shape[:2]
        ov_h, ov_w = img_overlay.shape[:2]
        pos_x, pos_y = int(pos_x), int(pos_y)
        x1, y1 = max(0, pos_x), max(0, pos_y)
        x2, y2 = min(canvas_w, pos_x + ov_w), min(canvas_h, pos_y + ov_h)
        if x2 <= x1 or y2 <= y1:
            return img
        ov_x1 = x1 - pos_x
        ov_y1 = y1 - pos_y
        overlay_rgb = img_overlay[ov_y1:ov_y1 + (y2 - y1), ov_x1:ov_x1 + (x2 - x1), :3]
        mask_crop = alpha_mask[ov_y1:ov_y1 + (y2 - y1), ov_x1:ov_x1 + (x2 - x1)] / 255.0
        mask_3ch = np.dstack([mask_crop] * 3)
        bg_crop = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = (overlay_rgb * mask_3ch + bg_crop * (1.0 - mask_3ch)).astype(np.uint8)
        return img

    # --- TÜNEL GÖLGE (V24'teki perspektif tunnel void) ---
    def generate_tunnel_void(self, w, h, obj_cx, obj_cy, thickness, ptype):
        screen_cx, screen_cy = self.width // 2, self.height // 2
        void_w = w - 2 * thickness
        void_h = h - 2 * thickness
        if void_w <= 0 or void_h <= 0:
            return None

        vec_x = screen_cx - obj_cx
        vec_y = screen_cy - obj_cy
        light_cx = np.clip((void_w // 2) + int(vec_x * 0.3), 0, void_w)
        light_cy = np.clip((void_h // 2) + int(vec_y * 0.3), 0, void_h)

        Y, X = np.ogrid[:void_h, :void_w]
        dist_map = np.sqrt((X - light_cx) ** 2 + (Y - light_cy) ** 2)
        max_dist = np.sqrt(void_w ** 2 + void_h ** 2)
        norm_dist = np.clip(dist_map / (max_dist * 0.6), 0, 1)

        base_opacity = random.uniform(0.80, 0.98)
        alpha_channel = (np.power(norm_dist, 0.7) * base_opacity * 255).astype(np.uint8)

        base_color = np.zeros((void_h, void_w, 3), dtype=np.uint8)
        noise = np.random.randint(0, 20, (void_h, void_w, 3))
        base_color = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        return cv2.merge([base_color[:, :, 0], base_color[:, :, 1],
                          base_color[:, :, 2], alpha_channel])

    # --- 3D DEPTH RENDER (ortak) ---
    def _render_depth(self, pallet, depth_cmds):
        for cmd in depth_cmds:
            if cmd['dist'] <= 0:
                continue
            nx_v = cmd['vec_x'] / cmd['dist']
            ny_v = cmd['vec_y'] / cmd['dist']
            tox = int(nx_v * cmd['depth_len'])
            toy = int(ny_v * cmd['depth_len'])
            body_color = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + random.randint(30, 60)
            steps = min(int(max(abs(tox), abs(toy))), 30)
            for s in range(steps, 0, -1):
                r = s / steps if steps > 0 else 0
                self.overlay_image_alpha(pallet, body_color,
                                         cmd['pos_x'] + int(tox * r),
                                         cmd['pos_y'] + int(toy * r),
                                         cmd['alpha'])

    def _render_faces(self, pallet, face_cmds, annotations):
        for cmd in face_cmds:
            self.overlay_image_alpha(pallet, cmd['sprite'], cmd['pos_x'], cmd['pos_y'],
                                     cmd['sprite'][:, :, 3])
            bx = (cmd['pos_x'] + cmd['sw'] / 2) / self.width
            by = (cmd['pos_y'] + cmd['sh'] / 2) / self.height
            annotations.append({
                'class_id': cmd['ptype_idx'],
                'bbox': [bx, by, cmd['sw'] / self.width, cmd['sh'] / self.height]
            })

    # ========================================
    # YÖNLENDİRİCİ (ROUTER)
    # ========================================
    def generate_pallet(self, ptype):
        pallet = self.get_random_background()
        annotations = []

        if ptype in SPECIAL_PAIRS_CONFIG:
            self._gen_paired(pallet, annotations, ptype)
        elif ptype in PACKAGE_PROFILES:
            self._gen_package(pallet, annotations, ptype)
        else:
            self._gen_scattered(pallet, annotations, ptype)

        # Hafif noise augmentation
        if random.random() < 0.3:
            noise = np.random.normal(0, 3, pallet.shape).astype(np.int16)
            pallet = np.clip(pallet.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return pallet, annotations

    # ========================================
    # STRATEJİ 1: 1524/1560 (V14 Paired + Void)
    # ========================================
    def _gen_paired(self, pallet, annotations, ptype):
        scale = self.width / 640.0
        contours_list = self.loader.profiles_multi[ptype]
        cls_id = CLASS_MAP[ptype]

        pile_count = random.randint(1, 2)
        depth_cmds, face_cmds = [], []
        occupied_rects = []

        for _ in range(pile_count):
            current_ptype = ptype
            # %50 ihtimalle diğer paired profili de koy
            if random.random() < 0.50:
                other = [s for s in SPECIAL_PAIRS_CONFIG.keys() if s != ptype]
                if other and other[0] in self.loader.profiles_multi:
                    current_ptype = random.choice(other)

            cur_contours = self.loader.profiles_multi[current_ptype]
            cur_cls_id = CLASS_MAP[current_ptype]

            if current_ptype == '1560':
                base_size = int(random.randint(100, 120) * scale)
            else:
                base_size = int(random.randint(50, 60) * scale)

            target_rotation = 0
            if current_ptype == '1560':
                if random.random() < 0.5:
                    target_rotation = 90
                    grid_cols = random.randint(1, 2)
                    grid_rows = random.randint(8, 14)
                else:
                    grid_cols = random.randint(5, 12)
                    grid_rows = random.randint(1, 4)
            elif current_ptype == '1524':
                shape_roll = random.random()
                if shape_roll < 0.3:
                    grid_cols, grid_rows = 1, random.randint(5, 12)
                elif shape_roll < 0.6:
                    grid_cols, grid_rows = random.randint(4, 10), 1
                else:
                    grid_cols, grid_rows = random.randint(3, 6), random.randint(3, 8)

            ref_sprite, ref_t = self.renderer.create_paired_block(
                cur_contours, base_size, current_ptype, target_rotation)
            ph, pw = ref_sprite.shape[:2]
            if ph == 0 or pw == 0:
                continue

            pile_w = grid_cols * pw
            pile_h = grid_rows * ph
            max_x = max(0, self.width - pile_w)
            max_y = max(0, self.height - pile_h)

            found_pos = False
            for _ in range(50):
                sx = random.randint(0, max_x)
                sy = random.randint(0, max_y)
                rect = [sx, sy, pile_w, pile_h]
                if not self._check_collision(rect, occupied_rects):
                    occupied_rects.append(rect)
                    found_pos = True
                    break
            if not found_pos:
                continue

            for r in range(grid_rows):
                cur_y = sy + r * ph
                for c in range(grid_cols):
                    cur_x = sx + c * pw
                    if cur_x + pw > self.width or cur_y + ph > self.height:
                        continue

                    sprite, t = self.renderer.create_paired_block(
                        cur_contours, base_size, current_ptype, target_rotation)
                    sh, sw = sprite.shape[:2]
                    jx = random.randint(-1, 1)
                    jy = random.randint(-1, 1)
                    fx, fy = cur_x + jx, cur_y + jy

                    obj_cx = fx + sw // 2
                    obj_cy = fy + sh // 2
                    vec_x = self.width // 2 - obj_cx
                    vec_y = self.height // 2 - obj_cy
                    dist = math.sqrt(vec_x ** 2 + vec_y ** 2)
                    max_dist = math.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
                    depth_len = min((dist / max_dist) * (50 * scale), sw * 0.8)

                    depth_cmds.append({
                        'pos_x': fx, 'pos_y': fy, 'sw': sw, 'sh': sh,
                        'vec_x': vec_x, 'vec_y': vec_y, 'dist': dist,
                        'depth_len': depth_len, 'alpha': sprite[:, :, 3]
                    })
                    face_cmds.append({
                        'sprite': sprite, 'pos_x': fx, 'pos_y': fy,
                        'sw': sw, 'sh': sh, 'ptype_idx': cur_cls_id
                    })

        self._render_depth(pallet, depth_cmds)
        self._render_faces(pallet, face_cmds, annotations)

    # ========================================
    # STRATEJİ 2: 9859 (Test Tuner Package + Pattern)
    # ========================================
    def _gen_package(self, pallet, annotations, ptype):
        cls_id = CLASS_MAP[ptype]
        profiles_per_pkg = PACKAGE_PROFILES[ptype]

        base_sprite = self.renderer.get_9859_sprite(150)
        if base_sprite is None:
            return

        pkg_img = self.renderer.create_package_9859(base_sprite, profiles_per_pkg)
        pkg_h, pkg_w = pkg_img.shape[:2]

        pattern = random.choices(
            ["ALTERNATING", "SANDWICH", "UNIFORM"],
            weights=[0.70, 0.20, 0.10], k=1)[0]
        num_rows = random.randint(8, 16)
        pkgs_per_row = random.randint(3, 5)

        pile_width = pkgs_per_row * pkg_w
        start_x = max(10, (self.width - pile_width) // 2)
        current_y = self.height - random.randint(80, 150)

        depth_cmds, face_cmds = [], []

        for row_idx in range(num_rows):
            row_type = "HORIZONTAL"
            flip = False

            if pattern == "ALTERNATING" and row_idx % 2 == 1:
                flip = True
            elif pattern == "SANDWICH" and (num_rows // 3 <= row_idx < 2 * (num_rows // 3)):
                row_type = "VERTICAL"

            if row_type == "HORIZONTAL":
                current_y -= pkg_h
                if current_y < 0:
                    break
                for col in range(pkgs_per_row):
                    x = start_x + col * pkg_w
                    if x + pkg_w > self.width:
                        break
                    stamp = cv2.rotate(pkg_img, cv2.ROTATE_180) if flip else pkg_img
                    self._queue_pkg_render(depth_cmds, face_cmds, stamp, x, current_y,
                                           cls_id, annotations, profiles_per_pkg, False)
            else:
                stamp = cv2.rotate(pkg_img, cv2.ROTATE_90_CLOCKWISE)
                current_y -= stamp.shape[0]
                if current_y < 0:
                    break
                v_pkgs = pile_width // stamp.shape[1] if stamp.shape[1] > 0 else 1
                offset_x = start_x + (pile_width - v_pkgs * stamp.shape[1]) // 2
                for col in range(v_pkgs):
                    x = offset_x + col * stamp.shape[1]
                    if x + stamp.shape[1] > self.width:
                        break
                    self._queue_pkg_render(depth_cmds, face_cmds, stamp, x, current_y,
                                           cls_id, annotations, profiles_per_pkg, True)

        # Depth + Void
        for cmd in depth_cmds:
            if cmd['dist'] > 0:
                nx_v = cmd['vec_x'] / cmd['dist']
                ny_v = cmd['vec_y'] / cmd['dist']
                tox = int(nx_v * cmd['depth_len'])
                toy = int(ny_v * cmd['depth_len'])
                body = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + random.randint(30, 60)
                steps = min(int(max(abs(tox), abs(toy))), 20)
                for s in range(steps, 0, -1):
                    r = s / steps if steps > 0 else 0
                    self.overlay_image_alpha(pallet, body,
                                             cmd['pos_x'] + int(tox * r),
                                             cmd['pos_y'] + int(toy * r),
                                             cmd['alpha'])
            if cmd.get('void_sprite') is not None:
                vs = cmd['void_sprite']
                self.overlay_image_alpha(pallet, vs, cmd['pos_x'] + 2, cmd['pos_y'] + 2, vs[:, :, 3])

        # Faces
        for cmd in face_cmds:
            self.overlay_image_alpha(pallet, cmd['sprite'], cmd['pos_x'], cmd['pos_y'],
                                     cmd['sprite'][:, :, 3])

    def _queue_pkg_render(self, depth_list, face_list, stamp, x, y,
                          cls_id, annotations, profiles_per_pkg, is_vertical):
        sh, sw = stamp.shape[:2]
        obj_cx = x + sw // 2
        obj_cy = y + sh // 2
        vec_x = (self.width // 2) - obj_cx
        vec_y = (self.height // 2) - obj_cy
        dist = math.sqrt(vec_x ** 2 + vec_y ** 2)
        max_dist = math.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        depth_len = min((dist / max_dist) * 60, sw * 0.8)

        # Tunnel void (Test Tuner'dan)
        void_w, void_h = sw - 4, sh - 4
        void_sprite = None
        if void_w > 0 and void_h > 0:
            light_cx = np.clip((void_w // 2) + int(vec_x * 0.3), 0, void_w)
            light_cy = np.clip((void_h // 2) + int(vec_y * 0.3), 0, void_h)
            Y, X = np.ogrid[:void_h, :void_w]
            dm = np.sqrt((X - light_cx) ** 2 + (Y - light_cy) ** 2)
            md = np.sqrt(void_w ** 2 + void_h ** 2)
            nd = np.clip(dm / (md * 0.6), 0, 1)
            alpha = (np.power(nd, 0.7) * random.uniform(0.80, 0.98) * 255).astype(np.uint8)
            base_color = np.zeros((void_h, void_w, 3), dtype=np.uint8)
            void_sprite = cv2.merge([base_color[:, :, 0], base_color[:, :, 1],
                                     base_color[:, :, 2], alpha])

        depth_list.append({
            'pos_x': x, 'pos_y': y, 'sw': sw, 'sh': sh, 'alpha': stamp[:, :, 3],
            'vec_x': vec_x, 'vec_y': vec_y, 'dist': dist, 'depth_len': depth_len,
            'void_sprite': void_sprite
        })
        face_list.append({
            'sprite': stamp, 'pos_x': x, 'pos_y': y, 'sw': sw, 'sh': sh
        })

        # 6x bounding box (her profil ayrı etiket)
        for i in range(profiles_per_pkg):
            if is_vertical:
                box_w = sw
                box_h = sh / float(profiles_per_pkg)
                bx = x
                by = y + i * box_h
            else:
                box_w = sw / float(profiles_per_pkg)
                box_h = sh
                bx = x + i * box_w
                by = y
            center_x = (bx + box_w / 2.0) / self.width
            center_y = (by + box_h / 2.0) / self.height
            annotations.append({
                'class_id': cls_id,
                'bbox': [center_x, center_y, box_w / self.width, box_h / self.height]
            })

    # ========================================
    # STRATEJİ 3: 7170/9794 (V24 Scattered + Tunnel Shadow)
    # ========================================
    def _gen_scattered(self, pallet, annotations, ptype):
        scale = self.width / 640.0
        cls_id = CLASS_MAP[ptype]
        contour = self.loader.profiles_single[ptype]
        base_size = int(random.randint(25, 40) * scale)

        ref_sprite, ref_t = self.renderer.get_single_half_single(contour, base_size, 0, ptype)
        if ref_sprite is None:
            return
        ph, pw = ref_sprite.shape[:2]
        if ph == 0 or pw == 0:
            return

        current_y = self.height - int(10 * scale)
        depth_cmds, face_cmds = [], []

        while current_y > int(50 * scale):
            spacer = random.randint(int(5 * scale), int(15 * scale))
            row_y = current_y - spacer - ph
            if row_y < 0:
                break

            current_x = random.randint(0, int(30 * scale))
            while current_x < self.width - pw:
                rot = random.choice([0, 180])
                sz = int(base_size * random.uniform(0.95, 1.05))
                sprite, t = self.renderer.get_single_half_single(contour, sz, rot, ptype)
                if sprite is None:
                    break
                sh, sw = sprite.shape[:2]

                pos_x = current_x
                pos_y = row_y + random.randint(-2, 2)
                if pos_x + sw > self.width:
                    break

                obj_cx = pos_x + sw // 2
                obj_cy = pos_y + sh // 2
                vec_x = self.width // 2 - obj_cx
                vec_y = self.height // 2 - obj_cy
                dist = math.sqrt(vec_x ** 2 + vec_y ** 2)
                max_dist = math.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
                depth_len = min((dist / max_dist) * (50 * scale), sw * 0.8)

                # Tunnel void (V24'teki gibi)
                void_sprite = self.generate_tunnel_void(sw, sh, obj_cx, obj_cy, t, ptype)

                depth_cmds.append({
                    'pos_x': pos_x, 'pos_y': pos_y, 'sw': sw, 'sh': sh,
                    'vec_x': vec_x, 'vec_y': vec_y, 'dist': dist,
                    'depth_len': depth_len, 'alpha': sprite[:, :, 3],
                    'void_sprite': void_sprite, 'thickness': t
                })
                face_cmds.append({
                    'sprite': sprite, 'pos_x': pos_x, 'pos_y': pos_y,
                    'sw': sw, 'sh': sh, 'ptype_idx': cls_id
                })

                gap = random.randint(int(2 * scale), int(5 * scale))
                current_x += (sw + gap)
            current_y = row_y

        # Depth + Void render
        for cmd in depth_cmds:
            if cmd['dist'] <= 0:
                continue
            nx_v = cmd['vec_x'] / cmd['dist']
            ny_v = cmd['vec_y'] / cmd['dist']
            tox = int(nx_v * cmd['depth_len'])
            toy = int(ny_v * cmd['depth_len'])
            body = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + 50
            steps = min(int(max(abs(tox), abs(toy))), 30)
            for s in range(steps, 0, -1):
                r = s / steps if steps > 0 else 0
                self.overlay_image_alpha(pallet, body,
                                         cmd['pos_x'] + int(tox * r),
                                         cmd['pos_y'] + int(toy * r),
                                         cmd['alpha'])
            # Void
            if cmd.get('void_sprite') is not None:
                vs = cmd['void_sprite']
                t = cmd.get('thickness', 2)
                self.overlay_image_alpha(pallet, vs, cmd['pos_x'] + t, cmd['pos_y'] + t, vs[:, :, 3])

        self._render_faces(pallet, face_cmds, annotations)

    def _check_collision(self, new_rect, occupied):
        nx_r, ny_r, nw, nh = new_rect
        for (ox, oy, ow, oh) in occupied:
            if (nx_r < ox + ow and nx_r + nw > ox and
                    ny_r < oy + oh and ny_r + nh > oy):
                return True
        return False


# ==========================================
# --- 5. CLAHE ---
# ==========================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


# ==========================================
# --- 6. ANA ÇALIŞTIRMA ---
# ==========================================
def main():
    loader = DXFProfileLoader()
    loader.load_all()

    available = [n for n in STATIC_PROFILES.keys() if loader.has_profile(n)]
    if not available:
        print("❌ Hiçbir profil yüklenemedi!")
        return

    print(f"\n📋 Yüklenen profiller: {available}")

    renderer = ProfileRenderer(loader)
    generator = SyntheticPalletGenerator(loader, renderer, BG_FOLDER)

    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    for subset in ['train', 'val']:
        os.makedirs(f'{BASE_DIR}/images/{subset}', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/labels/{subset}', exist_ok=True)

    print(f"\n🚀 Sentetik Veri V25 FINAL Başlıyor...")
    print(f"   1524/1560 -> Paired Block + Void Fill (V14)")
    print(f"   9859      -> Package + Pattern (Test Tuner)")
    print(f"   7170/9794 -> Scattered + Tunnel Shadow (V24)")

    for ptype in available:
        print(f"\n📦 Üretiliyor: {ptype} ...")

        for i in range(IMAGES_PER_CLASS_TRAIN):
            if i % 10 == 0:
                print(f"   [Train] {i}/{IMAGES_PER_CLASS_TRAIN}")
            pallet, anns = generator.generate_pallet(ptype)
            name = f"{ptype}_train_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/train/{name}.jpg', pallet)
            label_str = "".join([
                f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} "
                f"{a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n"
                for a in anns
            ])
            with open(f'{BASE_DIR}/labels/train/{name}.txt', 'w') as f:
                f.write(label_str)

        for i in range(IMAGES_PER_CLASS_VAL):
            if i % 5 == 0:
                print(f"   [Val] {i}/{IMAGES_PER_CLASS_VAL}")
            pallet, anns = generator.generate_pallet(ptype)
            name = f"{ptype}_val_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/val/{name}.jpg', pallet)
            label_str = "".join([
                f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} "
                f"{a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n"
                for a in anns
            ])
            with open(f'{BASE_DIR}/labels/val/{name}.txt', 'w') as f:
                f.write(label_str)

    # data.yaml
    with open(f'{BASE_DIR}/data.yaml', 'w') as f:
        names = "\n".join([f"  {i}: '{n}'" for i, n in enumerate(STATIC_PROFILES.keys())])
        f.write(
            f"path: {os.path.abspath(BASE_DIR)}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n{names}\n"
            f"nc: {len(STATIC_PROFILES)}"
        )

    print(f"\n✅ V25 FINAL TAMAMLANDI! (Klasör: {BASE_DIR})")


if __name__ == "__main__":
    main()