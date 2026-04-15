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
IMAGES_PER_CLASS_TRAIN = 5  # Her sınıf için üretilecek Train sayısı
IMAGES_PER_CLASS_VAL = 1     # Her sınıf için üretilecek Val sayısı
IMG_WIDTH = 1280
IMG_HEIGHT = 1280

BASE_DIR = "dataset_5_profile_FINAL" 
BG_FOLDER = "background"
DXF_FOLDER = "cad_files"

# SADECE BU 5 DOSYAYA BAKILACAK (Bozuk DXF'leri atlar)
STATIC_PROFILES = {
    '1524': os.path.join(DXF_FOLDER, "1524_CLEAN.dxf"),
    '1560': os.path.join(DXF_FOLDER, "1560_CLEAN.dxf"),
    '9859': os.path.join(DXF_FOLDER, "9859.dxf"),
    '7170': os.path.join(DXF_FOLDER, "7170.dxf"),
    '9794': os.path.join(DXF_FOLDER, "9794.dxf")
}

# Özel Sıkılık Ayarları (Script 1'den)
SPECIAL_PAIRS_CONFIG = {
    '1524': 0.70,  # Kare
    '1560': 0.85   # Dikdörtgen (Sıfır Boşluk)
}

# ==========================================
# --- 2. DXF YÜKLEYİCİ (İKİ FARKLI MANTIK) ---
# ==========================================
class DXFProfileLoader:
    def __init__(self): 
        self.profiles = {}       # 1524, 1560, 7170, 9794 için (Graph tabanlı)
        self.lines_9859 = None   # 9859 için özel (Line tabanlı)

    def load_all(self):
        for name, path in STATIC_PROFILES.items():
            if not os.path.exists(path):
                print(f"⚠️ HATA: {path} bulunamadı! Bu profil atlanacak.")
                continue
            
            if name == '9859':
                self._load_9859_lines(path)
            else:
                self._load_standard_graph(path, name)

    def _load_standard_graph(self, dxf_path, profile_name):
        # SCRIPT 1 ve 3'teki standart Graph/Contour mantığı
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            G = nx.Graph()
            def to_key(p): return (round(p[0], 3), round(p[1], 3))
            
            for e in msp:
                if e.dxftype() == 'LINE': G.add_edge(to_key(e.dxf.start), to_key(e.dxf.end))
                elif e.dxftype() == 'LWPOLYLINE':
                    pts = e.get_points('xy')
                    for i in range(len(pts)-1): G.add_edge(to_key(pts[i]), to_key(pts[i+1]))
                    if e.closed: G.add_edge(to_key(pts[-1]), to_key(pts[0]))
            
            components = list(nx.connected_components(G))
            if not components: return
            largest_comp = max(components, key=len)
            subgraph = G.subgraph(largest_comp)
            ordered_nodes = list(nx.dfs_preorder_nodes(subgraph))
            
            pts = np.array(ordered_nodes)
            min_vals, max_vals = np.min(pts, axis=0), np.max(pts, axis=0)
            center = (min_vals + max_vals) / 2
            pts = pts - center
            h = max_vals[1] - min_vals[1]
            self.profiles[profile_name] = pts / h if h > 0 else pts
            print(f"✅ Yüklendi: {profile_name}")
        except Exception as e:
            print(f"❌ DXF Hatası ({profile_name}): {e}")

    def _load_9859_lines(self, dxf_path):
        # SCRIPT 2'deki 9859'a özel açık/parçalı line mantığı
        try:
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
            
            if not lines: return
            
            all_pts = np.array([pt for p1, p2 in lines for pt in (p1, p2)])
            min_vals, max_vals = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
            max_dim = max(max_vals[0] - min_vals[0], max_vals[1] - min_vals[1])
            if max_dim == 0: max_dim = 1
            
            norm_lines = []
            for p1, p2 in lines:
                x1, y1 = (p1[0] - min_vals[0]) / max_dim, (p1[1] - min_vals[1]) / max_dim
                x2, y2 = (p2[0] - min_vals[0]) / max_dim, (p2[1] - min_vals[1]) / max_dim
                norm_lines.append(((x1, y1), (x2, y2)))
            
            self.lines_9859 = norm_lines
            print("✅ Yüklendi: 9859 (Özel Paket Mantığı)")
        except Exception as e:
            print(f"❌ DXF Hatası (9859): {e}")

# ==========================================
# --- 3. RENDER MOTORU ---
# ==========================================
class ProfileRenderer:
    def __init__(self, loader):
        self.loader = loader

    def apply_aluminum_texture(self, mask, profile_name=""):
        h, w = mask.shape
        # 9859 için Script 2'deki parlaklık mantığı
        if profile_name == '9859':
            is_shiny = random.random() < 0.3
            base_val = random.randint(200, 240) if is_shiny else random.randint(150, 190)
            contrast = 30 if is_shiny else 15
        else: # Diğerleri
            is_shiny = random.random() < 0.30
            base_val = random.randint(230, 255) if is_shiny else random.randint(180, 220)
            contrast = 40 if is_shiny else 25

        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        if random.random() < 0.5:
            for y in range(h):
                v = int(np.clip(base_val - contrast * np.sin(y / h * 3.14), 0, 255))
                img_bgr[y, :, :] = (v, v, v)
        else:
            for x in range(w):
                v = int(np.clip(base_val - contrast * np.sin(x / w * 3.14), 0, 255))
                img_bgr[:, x, :] = (v, v, v)
        
        noise = np.random.randint(-15, 15, img_bgr.shape)
        img_bgr = np.clip(img_bgr.astype(int) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, (50, 50, 50), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    # --- 9859 ÖZEL RENDER (SCRIPT 2) ---
    def get_9859_sprite(self, size):
        if not self.loader.lines_9859: return None
        canvas_size = int(size * 3.0)
        
        padding = int(canvas_size * 0.1)
        scale = (canvas_size - 2 * padding)
        
        # Çizgileri büyüt ve Poligona çevir
        scaled_lines = []
        for p1, p2 in self.loader.lines_9859:
            x1, y1 = p1[0] * scale + padding, canvas_size - (p1[1] * scale + padding)
            x2, y2 = p2[0] * scale + padding, canvas_size - (p2[1] * scale + padding)
            scaled_lines.append((np.array([x1, y1]), np.array([x2, y2])))
            
        polygons = []
        edges = list(scaled_lines)
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
                if best_idx != -1 and best_dist <= 3.0:
                    edge = edges.pop(best_idx)
                    current_poly.append(edge[0] if best_reverse else edge[1])
                else: break
            if len(current_poly) > 2: polygons.append(np.array(current_poly, dtype=np.int32))
                
        polygons = sorted(polygons, key=cv2.contourArea, reverse=True)
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        for poly in polygons:
            temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [poly], 255)
            mask = cv2.bitwise_xor(mask, temp_mask)
            
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(mask)
        if coords is None: return None
        
        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y+h, x:x+w]
        
        textured_sprite = self.apply_aluminum_texture(cropped_mask, '9859')
        if textured_sprite.shape[1] > textured_sprite.shape[0]:
            textured_sprite = cv2.rotate(textured_sprite, cv2.ROTATE_90_CLOCKWISE)
        return textured_sprite

    def create_package_9859(self, sprite, count=6):
        h, w = sprite.shape[:2]
        pkg = np.zeros((h, w * count, 4), dtype=np.uint8)
        for i in range(count):
            pkg[:, i*w : (i+1)*w] = sprite
        return pkg

    # --- STANDART VE ÇİFTLİ RENDER (SCRIPT 1/3) ---
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
        cropped_mask = mask[y:y+h, x:x+w]
        thickness = int(min(w, h) * 0.15) 
        return self.apply_aluminum_texture(cropped_mask), thickness

    def create_paired_block(self, contour, size, profile_name, target_rotation=0):
        sprite_a, t = self.get_single_half(contour, size, 0)
        if sprite_a is None: return np.zeros((10,10,4), dtype=np.uint8), 0
        h, w = sprite_a.shape[:2]
        sprite_b_raw, _ = self.get_single_half(contour, size, 180)
        sprite_b = cv2.resize(sprite_b_raw, (w, h))
        
        tightness = SPECIAL_PAIRS_CONFIG.get(profile_name, 1.0)
        shift_val = int(t * tightness)
        
        canvas_w = w * 2 + shift_val * 4
        canvas_h = h * 2 + shift_val * 4
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        cx, cy = canvas_w // 2, canvas_h // 2
        
        pos_a_x, pos_a_y = cx - w // 2, cy - h // 2
        pos_b_x, pos_b_y = pos_a_x + shift_val, pos_a_y + shift_val
        
        def paste(bg, fg, x, y):
            fh, fw = fg.shape[:2]
            if x+fw > bg.shape[1] or y+fh > bg.shape[0]: return bg
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                bg[y:y+fh, x:x+fw, c] = (fg[:,:,c] * alpha + bg[y:y+fh, x:x+fw, c] * (1-alpha))
            bg[y:y+fh, x:x+fw, 3] = np.maximum(bg[y:y+fh, x:x+fw, 3], fg[:,:,3])
            return bg

        canvas = paste(canvas, sprite_a, pos_a_x, pos_a_y)
        canvas = paste(canvas, sprite_b, pos_b_x, pos_b_y)
        
        if target_rotation == 90 or target_rotation == 270:
            canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

        coords = cv2.findNonZero(canvas[:, :, 3])
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            return canvas[y:y+h_crop, x:x+w_crop], t
        return canvas, 0

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
        self.class_map = {name: i for i, name in enumerate(STATIC_PROFILES.keys())}

    def get_random_background(self):
        if self.bg_images and random.random() < 0.95: 
            bg_path = random.choice(self.bg_images)
            bg = cv2.imread(bg_path)
            if bg is not None:
                bg = cv2.resize(bg, (self.width, self.height))
                return (bg.astype(float) * random.uniform(0.5, 0.9)).astype(np.uint8)
        val = random.randint(40, 100)
        bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * val
        noise = np.random.randint(-30, 30, bg.shape)
        return np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)

    def overlay_image_alpha(self, img, img_overlay, pos_x, pos_y, alpha_mask):
        canvas_h, canvas_w = img.shape[:2]
        ov_h, ov_w = img_overlay.shape[:2]
        x1, y1 = max(0, int(pos_x)), max(0, int(pos_y))
        x2, y2 = min(canvas_w, int(pos_x) + ov_w), min(canvas_h, int(pos_y) + ov_h)
        if x2 <= x1 or y2 <= y1: return img
        ov_x1, ov_y1 = x1 - int(pos_x), y1 - int(pos_y)
        ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)
        
        overlay_rgb = img_overlay[ov_y1:ov_y2, ov_x1:ov_x2, :3]
        mask_3ch = np.dstack([alpha_mask[ov_y1:ov_y2, ov_x1:ov_x2] / 255.0] * 3)
        bg_crop = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = (overlay_rgb * mask_3ch + bg_crop * (1.0 - mask_3ch)).astype(np.uint8)
        return img

    def generate_tunnel_void(self, w, h, obj_cx, obj_cy, thickness, ptype):
        screen_cx, screen_cy = self.width // 2, self.height // 2
        void_w, void_h = w - 2 * thickness, h - 2 * thickness
        if void_w <= 0 or void_h <= 0: return None
        
        vec_x, vec_y = screen_cx - obj_cx, screen_cy - obj_cy
        light_center_x = np.clip((void_w // 2) + int(vec_x * 0.3), 0, void_w)
        light_center_y = np.clip((void_h // 2) + int(vec_y * 0.3), 0, void_h)
        
        Y, X = np.ogrid[:void_h, :void_w]
        dist_map = np.sqrt((X - light_center_x)**2 + (Y - light_center_y)**2)
        max_dist = np.sqrt(void_w**2 + void_h**2)
        norm_dist = np.clip(dist_map / (max_dist * 0.6), 0, 1)
        
        base_opacity = random.uniform(0.92, 0.99) if ptype == '1524' else random.uniform(0.80, 0.95)
        alpha_channel = (np.power(norm_dist, 0.7) * base_opacity * 255).astype(np.uint8)
        
        base_color = np.zeros((void_h, void_w, 3), dtype=np.uint8)
        noise = np.random.randint(0, 20, (void_h, void_w, 3))
        base_color = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        return cv2.merge([base_color[:,:,0], base_color[:,:,1], base_color[:,:,2], alpha_channel])

    # YÖNLENDİRİCİ (ROUTER)
    def generate_pallet(self, ptype):
        pallet = self.get_random_background()
        annotations = []
        
        if ptype in SPECIAL_PAIRS_CONFIG:
            self._gen_1524_1560(pallet, annotations, ptype)
        elif ptype == '9859':
            self._gen_9859(pallet, annotations)
        else: # 7170, 9794
            self._gen_7170_9794(pallet, annotations, ptype)
            
        return pallet, annotations

    # --- STRATEJİ 1: 1524 / 1560 (PAIRED & PITCH BLACK) ---
    def _gen_1524_1560(self, pallet, annotations, ptype):
        scale = self.width / 640.0
        contour = self.loader.profiles[ptype]
        cls_id = self.class_map[ptype]
        
        base_size = int(random.randint(100, 120) * scale) if ptype == '1560' else int(random.randint(50, 60) * scale)
        
        target_rotation = 0
        if ptype == '1560':
            if random.random() < 0.5: target_rotation = 90; cols, rows = random.randint(1, 2), random.randint(8, 14)
            else: target_rotation = 0; cols, rows = random.randint(5, 12), random.randint(1, 4)
        else:
            shape_roll = random.random()
            if shape_roll < 0.3: cols, rows = 1, random.randint(5, 12)
            elif shape_roll < 0.6: cols, rows = random.randint(4, 10), 1
            else: cols, rows = random.randint(3, 6), random.randint(3, 8)

        ref_sprite, t = self.renderer.create_paired_block(contour, base_size, ptype, target_rotation)
        ph, pw = ref_sprite.shape[:2]
        if ph == 0: return

        pile_w, pile_h = cols * pw, rows * ph
        start_x = random.randint(0, max(0, self.width - pile_w))
        start_y = random.randint(0, max(0, self.height - pile_h))

        depth_cmds, face_cmds = [], []
        
        for r in range(rows):
            for c in range(cols):
                fx = start_x + c * pw + random.randint(-1, 1)
                fy = start_y + r * ph + random.randint(-1, 1)
                
                sprite, t = self.renderer.create_paired_block(contour, base_size, ptype, target_rotation)
                sh, sw = sprite.shape[:2]
                
                cx, cy = self.width // 2, self.height // 2
                obj_cx, obj_cy = fx + sw // 2, fy + sh // 2
                vec_x, vec_y = cx - obj_cx, cy - obj_cy
                dist = math.sqrt(vec_x**2 + vec_y**2)
                max_dist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                depth_len = min((dist/max_dist)*(50*scale), sw*0.8)
                
                void = self.generate_tunnel_void(sw, sh, obj_cx, obj_cy, t, ptype)
                if void is not None: self.overlay_image_alpha(pallet, void, fx+t, fy+t, void[:,:,3])

                depth_cmds.append({'x': fx, 'y': fy, 'sw': sw, 'sh': sh, 'vx': vec_x, 'vy': vec_y, 'd': dist, 'dl': depth_len, 'alpha': sprite[:,:,3]})
                face_cmds.append({'sprite': sprite, 'x': fx, 'y': fy, 'sw': sw, 'sh': sh})

        for cmd in depth_cmds:
            if cmd['d'] > 0:
                tox, toy = int((cmd['vx']/cmd['d'])*cmd['dl']), int((cmd['vy']/cmd['d'])*cmd['dl'])
                body = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + 50
                steps = min(int(max(abs(tox), abs(toy))), 30)
                for s in range(steps, 0, -1):
                    r = s/steps if steps>0 else 0
                    self.overlay_image_alpha(pallet, body, cmd['x']+int(tox*r), cmd['y']+int(toy*r), cmd['alpha'])
                    
        for cmd in face_cmds:
            self.overlay_image_alpha(pallet, cmd['sprite'], cmd['x'], cmd['y'], cmd['sprite'][:,:,3])
            bx, by = (cmd['x']+cmd['sw']/2)/self.width, (cmd['y']+cmd['sh']/2)/self.height
            annotations.append({'class_id': cls_id, 'bbox': [bx, by, cmd['sw']/self.width, cmd['sh']/self.height]})

    # --- STRATEJİ 2: 9859 (PACKAGE & PATTERNS & 6x BBOX) ---
    def _gen_9859(self, pallet, annotations):
        cls_id = self.class_map['9859']
        base_sprite = self.renderer.get_9859_sprite(150) # Boyut sabit (Test kodundaki gibi)
        if base_sprite is None: return
        
        pkg_img = self.renderer.create_package_9859(base_sprite, 6)
        pkg_h, pkg_w = pkg_img.shape[:2]
        
        pattern = random.choices(["ALTERNATING", "SANDWICH", "UNIFORM"], weights=[0.70, 0.20, 0.10], k=1)[0]
        num_rows = random.randint(8, 16)
        pkgs_per_row = random.randint(3, 5)
        
        pile_width = pkgs_per_row * pkg_w
        start_x = (self.width - pile_width) // 2
        current_y = self.height - random.randint(80, 150)
        
        depth_cmds, face_cmds = [], []
        
        for row_idx in range(num_rows):
            row_type = "HORIZONTAL"
            flip = False
            
            if pattern == "ALTERNATING" and row_idx % 2 == 1: flip = True
            elif pattern == "SANDWICH" and (num_rows // 3 <= row_idx < 2 * (num_rows // 3)): row_type = "VERTICAL"
            
            if row_type == "HORIZONTAL":
                current_y -= pkg_h 
                for col in range(pkgs_per_row):
                    x = start_x + (col * pkg_w)
                    stamp = cv2.rotate(pkg_img, cv2.ROTATE_180) if flip else pkg_img
                    self._queue_9859_render(depth_cmds, face_cmds, stamp, x, current_y, cls_id, annotations, is_vertical=False)
            else:
                stamp = cv2.rotate(pkg_img, cv2.ROTATE_90_CLOCKWISE)
                current_y -= stamp.shape[0]
                v_pkgs = pile_width // stamp.shape[1]
                offset_x = start_x + (pile_width - (v_pkgs * stamp.shape[1])) // 2
                for col in range(v_pkgs):
                    x = offset_x + (col * stamp.shape[1])
                    self._queue_9859_render(depth_cmds, face_cmds, stamp, x, current_y, cls_id, annotations, is_vertical=True)

        # 3D Depth
        for cmd in depth_cmds:
            if cmd['d'] > 0:
                tox, toy = int((cmd['vx']/cmd['d'])*cmd['dl']), int((cmd['vy']/cmd['d'])*cmd['dl'])
                body = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + random.randint(30, 60)
                steps = min(int(max(abs(tox), abs(toy))), 20)
                for s in range(steps, 0, -1):
                    r = s/steps if steps>0 else 0
                    self.overlay_image_alpha(pallet, body, cmd['x']+int(tox*r), cmd['y']+int(toy*r), cmd['alpha'])
            if cmd['void'] is not None:
                self.overlay_image_alpha(pallet, cmd['void'], cmd['x']+2, cmd['y']+2, cmd['void'][:,:,3])

        # Face
        for cmd in face_cmds:
            self.overlay_image_alpha(pallet, cmd['sprite'], cmd['x'], cmd['y'], cmd['sprite'][:,:,3])

    def _queue_9859_render(self, depth_list, face_list, stamp, x, y, cls_id, annotations, is_vertical):
        sh, sw = stamp.shape[:2]
        obj_cx, obj_cy = x + sw // 2, y + sh // 2
        vec_x, vec_y = (self.width // 2) - obj_cx, (self.height // 2) - obj_cy
        dist = math.sqrt(vec_x**2 + vec_y**2)
        max_dist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
        depth_len = min((dist/max_dist)*60, sw*0.8)
        
        # Paket Gölgesi
        void_w, void_h = sw - 4, sh - 4
        if void_w > 0 and void_h > 0:
            Y, X = np.ogrid[:void_h, :void_w]
            dm = np.sqrt((X - void_w//2)**2 + (Y - void_h//2)**2)
            alpha = (np.clip(dm / (math.sqrt(void_w**2+void_h**2)*0.6), 0, 1)**0.7 * random.uniform(0.8, 0.98) * 255).astype(np.uint8)
            void_sprite = cv2.merge([np.zeros((void_h, void_w, 3), dtype=np.uint8), alpha])
        else: void_sprite = None
        
        depth_list.append({'x': x, 'y': y, 'sw': sw, 'sh': sh, 'alpha': stamp[:,:,3], 'vx': vec_x, 'vy': vec_y, 'd': dist, 'dl': depth_len, 'void': void_sprite})
        face_list.append({'sprite': stamp, 'x': x, 'y': y})

        # --- YAPAY ZEKA İÇİN KUSURSUZ 6x BOUNDING BOX ---
        # 1 paket = 6 profil. Her birini ayrı ayrı etiketliyoruz!
        for i in range(6):
            if is_vertical:
                box_w, box_h = sw, sh / 6.0
                bx = x
                by = y + i * box_h
            else:
                box_w, box_h = sw / 6.0, sh
                bx = x + i * box_w
                by = y
            
            # YOLO formatı (Center X, Center Y, Width, Height) normalize edilmiş
            center_x = (bx + box_w / 2.0) / self.width
            center_y = (by + box_h / 2.0) / self.height
            norm_w = box_w / self.width
            norm_h = box_h / self.height
            annotations.append({'class_id': cls_id, 'bbox': [center_x, center_y, norm_w, norm_h]})

    # --- STRATEJİ 3: 7170 / 9794 (DAĞINIK + STANDART GÖLGE) ---
    def _gen_7170_9794(self, pallet, annotations, ptype):
        scale = self.width / 640.0
        cls_id = self.class_map[ptype]
        contour = self.loader.profiles[ptype]
        base_size = int(random.randint(25, 40) * scale)
        
        ref_sprite, t = self.renderer.get_single_half(contour, base_size, 0)
        ph, pw = ref_sprite.shape[:2]
        if ph == 0: return

        current_y = self.height - int(10 * scale)
        depth_cmds, face_cmds = [], []

        while current_y > int(50 * scale):
            spacer = random.randint(int(5*scale), int(15*scale))
            row_y = current_y - spacer - ph
            if row_y < 0: break
            
            current_x = random.randint(0, int(30*scale))
            while current_x < self.width - pw:
                rot = random.choice([0, 180])
                sz = int(base_size * random.uniform(0.95, 1.05))
                sprite, _ = self.renderer.get_single_half(contour, sz, rot)
                sh, sw = sprite.shape[:2]
                
                pos_x, pos_y = current_x, row_y + random.randint(-2, 2)
                if pos_x + sw > self.width: break
                
                cx, cy = self.width // 2, self.height // 2
                obj_cx, obj_cy = pos_x + sw // 2, pos_y + sh // 2
                vec_x, vec_y = cx - obj_cx, cy - obj_cy
                dist = math.sqrt(vec_x**2 + vec_y**2)
                max_dist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                depth_len = min((dist/max_dist)*(50*scale), sw*0.8)

                depth_cmds.append({'x': pos_x, 'y': pos_y, 'sw': sw, 'sh': sh, 'vx': vec_x, 'vy': vec_y, 'd': dist, 'dl': depth_len, 'alpha': sprite[:,:,3]})
                face_cmds.append({'sprite': sprite, 'x': pos_x, 'y': pos_y, 'sw': sw, 'sh': sh})
                
                current_x += (sw + random.randint(int(2*scale), int(5*scale)))
            current_y = row_y

        for cmd in depth_cmds:
            if cmd['d'] > 0:
                tox, toy = int((cmd['vx']/cmd['d'])*cmd['dl']), int((cmd['vy']/cmd['d'])*cmd['dl'])
                body = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + 50
                steps = min(int(max(abs(tox), abs(toy))), 30)
                for s in range(steps, 0, -1):
                    r = s/steps if steps>0 else 0
                    self.overlay_image_alpha(pallet, body, cmd['x']+int(tox*r), cmd['y']+int(toy*r), cmd['alpha'])
                    
        for cmd in face_cmds:
            self.overlay_image_alpha(pallet, cmd['sprite'], cmd['x'], cmd['y'], cmd['sprite'][:,:,3])
            bx, by = (cmd['x']+cmd['sw']/2)/self.width, (cmd['y']+cmd['sh']/2)/self.height
            annotations.append({'class_id': cls_id, 'bbox': [bx, by, cmd['sw']/self.width, cmd['sh']/self.height]})

# ==========================================
# --- 5. ANA ÇALIŞTIRMA (MAIN) ---
# ==========================================
def main():
    loader = DXFProfileLoader()
    loader.load_all()

    if not loader.profiles and not loader.lines_9859:
        print("❌ Hiçbir profil yüklenemedi! CAD dosyalarını kontrol et.")
        return

    renderer = ProfileRenderer(loader)
    generator = SyntheticPalletGenerator(loader, renderer, BG_FOLDER)
    
    if os.path.exists(BASE_DIR): shutil.rmtree(BASE_DIR)
    for subset in ['train', 'val']:
        os.makedirs(f'{BASE_DIR}/images/{subset}', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/labels/{subset}', exist_ok=True)
        
    print(f"\n🚀 Sentetik Veri V25 (UNIFIED MULTI-STRATEGY) Başlıyor...")
    
    for ptype in STATIC_PROFILES.keys():
        if ptype not in loader.profiles and ptype != '9859': continue
        if ptype == '9859' and not loader.lines_9859: continue

        print(f"\n📦 Üretiliyor: {ptype} ...")
        
        for i in range(IMAGES_PER_CLASS_TRAIN):
            if i % 10 == 0: print(f"   [Train] {i}/{IMAGES_PER_CLASS_TRAIN}")
            pallet, anns = generator.generate_pallet(ptype)
            name = f"{ptype}_train_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/train/{name}.jpg', pallet)
            label_str = "".join([f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} {a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n" for a in anns])
            with open(f'{BASE_DIR}/labels/train/{name}.txt', 'w') as f: f.write(label_str)

        for i in range(IMAGES_PER_CLASS_VAL):
            if i % 5 == 0: print(f"   [Val] {i}/{IMAGES_PER_CLASS_VAL}")
            pallet, anns = generator.generate_pallet(ptype)
            name = f"{ptype}_val_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/val/{name}.jpg', pallet)
            label_str = "".join([f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} {a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n" for a in anns])
            with open(f'{BASE_DIR}/labels/val/{name}.txt', 'w') as f: f.write(label_str)

    # Data.yaml Oluştur
    with open(f'{BASE_DIR}/data.yaml', 'w') as f:
        names = "\n".join([f"  {i}: {n}" for i, n in enumerate(STATIC_PROFILES.keys())])
        f.write(f"path: {os.path.abspath(BASE_DIR)}\ntrain: images/train\nval: images/val\nnames:\n{names}\nnc: {len(STATIC_PROFILES)}")

    print(f"\n✅ V25 TAMAMLANDI! (Klasör: {BASE_DIR})")

if __name__ == "__main__":
    main()