import ezdxf
import cv2
import numpy as np
import random
import os
import math
import glob
import shutil
import networkx as nx

# --- AYARLAR ---
IMAGE_COUNT = 300            
IMG_WIDTH = 1280            
IMG_HEIGHT = 1280           
BASE_DIR = "new_test_dataset/dataset_v6"
BG_FOLDER = "background"
DXF_FOLDER = "cad_files"
FALSE_SAMPLES_DIR = "false_samples"

# Özel Profil Ayarları (Sıkılık Çarpanları)
SPECIAL_PAIRS_CONFIG = {
    '1524': 0.70,  # Kare (Sıkı - İç içe)
    '1560': 0.85   # Dikdörtgen (Biraz daha sıkılaştırdım ki boşluk hiç kalmasın)
}

# --- 1. DXF YÜKLEYİCİ ---
class DXFProfileLoader:
    def __init__(self): self.profiles = {}
    
    def load_dxf(self, dxf_path, profile_name):
        if not os.path.exists(dxf_path): return None
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            G = nx.Graph()
            def to_key(p): return (round(p[0], 3), round(p[1], 3))
            has_data = False
            for e in msp:
                if e.dxftype() == 'LINE':
                    G.add_edge(to_key(e.dxf.start), to_key(e.dxf.end)); has_data=True
                elif e.dxftype() == 'LWPOLYLINE':
                    pts = e.get_points('xy')
                    for i in range(len(pts)-1): G.add_edge(to_key(pts[i]), to_key(pts[i+1]))
                    if e.closed: G.add_edge(to_key(pts[-1]), to_key(pts[0]))
                    has_data=True
            
            if not has_data: return None
            components = list(nx.connected_components(G))
            largest_comp = max(components, key=len)
            subgraph = G.subgraph(largest_comp)
            ordered_nodes = list(nx.dfs_preorder_nodes(subgraph))
            
            contour = self.create_closed_contour(ordered_nodes)
            self.profiles[profile_name] = contour
            return contour
        except Exception as e:
            print(f"DXF Hatası ({profile_name}): {e}"); return None

    def create_closed_contour(self, points):
        pts = np.array(points)
        min_vals, max_vals = np.min(pts, axis=0), np.max(pts, axis=0)
        center = (min_vals + max_vals) / 2
        pts = pts - center
        h = max_vals[1] - min_vals[1]
        return pts / h if h > 0 else pts

# --- 2. RENDER MOTORU ---
class ProfileRenderer:
    def __init__(self): pass
    
    def apply_aluminum_texture(self, mask):
        h, w = mask.shape
        base_val = random.randint(180, 220)
        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        
        if random.random() < 0.5:
            for y in range(h):
                v = int(base_val - 25 * np.sin(y / h * 3.14))
                img_bgr[y, :, :] = (v, v, v)
        else:
            for x in range(w):
                v = int(base_val - 25 * np.sin(x / w * 3.14))
                img_bgr[:, x, :] = (v, v, v)
        
        noise = np.random.randint(-15, 15, img_bgr.shape)
        img_bgr = np.clip(img_bgr.astype(int) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, (60, 60, 60), 1, cv2.LINE_AA)
        
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
        cropped_mask = mask[y:y+h, x:x+w]
        thickness = int(w * 0.12)
        return self.apply_aluminum_texture(cropped_mask), thickness

    def create_paired_block(self, contour, size, profile_name, target_rotation=0):
        # --- ADIM 1: STANDART DİKEY MONTAJ (BOZULMAMASI İÇİN) ---
        # Her zaman rotasyon 0 ile üretip, en son çevireceğiz.
        
        sprite_a, t = self.get_single_half(contour, size, 0)
        if sprite_a is None: return np.zeros((10,10,4), dtype=np.uint8), (0,0,0,0)
        h, w = sprite_a.shape[:2]
        
        # Parça B (Tersi)
        sprite_b_raw, _ = self.get_single_half(contour, size, 180)
        sprite_b = cv2.resize(sprite_b_raw, (w, h))
        
        tightness = SPECIAL_PAIRS_CONFIG.get(profile_name, 1.0)
        shift_val = int(t * tightness)
        
        canvas_w = w * 2 + shift_val * 4
        canvas_h = h * 2 + shift_val * 4
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        cx, cy = canvas_w // 2, canvas_h // 2
        
        pos_a_x = cx - w // 2
        pos_a_y = cy - h // 2
        pos_b_x = pos_a_x + shift_val
        pos_b_y = pos_a_y + shift_val
        
        # --- VOID FILLING (NOISY GREYSCALE) ---
        void_x1 = min(pos_a_x, pos_b_x) + t
        void_y1 = min(pos_a_y, pos_b_y) + t
        void_x2 = max(pos_a_x + w, pos_b_x + w) - t
        void_y2 = max(pos_a_y + h, pos_b_y + h) - t
        
        if void_x2 > void_x1 and void_y2 > void_y1:
            vw = void_x2 - void_x1
            vh = void_y2 - void_y1
            
            # Renk Ayarı
            if profile_name == '1560':
                # Biraz aydınlık ama GRİ (Renk yok) + Noise
                base_val = np.random.randint(50, 90, (vh, vw), dtype=np.uint8)
            else:
                # 1524: Karanlık
                base_val = np.random.randint(10, 30, (vh, vw), dtype=np.uint8)
            
            void_img_bgr = cv2.merge([base_val, base_val, base_val])
            noise = np.random.randint(-20, 20, (vh, vw, 3))
            void_img_bgr = np.clip(void_img_bgr.astype(int) + noise, 0, 255).astype(np.uint8)
            
            center_v = (vw // 2, vh // 2)
            max_dist = math.sqrt((vw/2)**2 + (vh/2)**2)
            for vy in range(vh):
                for vx in range(vw):
                    dist = math.sqrt((vx - center_v[0])**2 + (vy - center_v[1])**2)
                    factor = 1.0 - (dist / max_dist) * 0.8
                    void_img_bgr[vy, vx] = (void_img_bgr[vy, vx] * factor).astype(np.uint8)

            void_img = cv2.merge([void_img_bgr[:,:,0], void_img_bgr[:,:,1], void_img_bgr[:,:,2], np.full((vh, vw), 255, dtype=np.uint8)])
            canvas[void_y1:void_y1+vh, void_x1:void_x1+vw] = void_img

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
        
        # --- ADIM 2: EĞER YATAY İSTENİYORSA ŞİMDİ ÇEVİR ---
        # Bu aşamada "Perfect Fit" bozulmaz, sadece resim döner.
        if target_rotation == 90 or target_rotation == 270:
            canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

        coords = cv2.findNonZero(canvas[:, :, 3])
        if coords is not None:
            x, y, w_crop, h_crop = cv2.boundingRect(coords)
            return canvas[y:y+h_crop, x:x+w_crop], (x, y, w_crop, h_crop)
        return canvas, (0,0,0,0)

    def render_profile_sprite(self, contour, size, rotation=0, profile_name="default"):
        if profile_name in SPECIAL_PAIRS_CONFIG:
            # Pair modunda 'rotation' parametresi tüm bloğun dönüşünü belirler
            return self.create_paired_block(contour, size, profile_name, target_rotation=rotation)
        
        sprite, t = self.get_single_half(contour, size, rotation)
        if sprite is None: return np.zeros((10,10,4), dtype=np.uint8), (0,0,0,0)
        return sprite, (0,0,sprite.shape[1], sprite.shape[0])

# --- 3. PALET ÜRETİCİ ---
class SyntheticPalletGenerator:
    def __init__(self, profile_loader, renderer, bg_folder='background'):
        self.loader = profile_loader
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
        bg = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)
        return bg

    def overlay_image_alpha(self, img, img_overlay, pos_x, pos_y, alpha_mask):
        canvas_h, canvas_w = img.shape[:2]
        ov_h, ov_w = img_overlay.shape[:2]
        pos_x, pos_y = int(pos_x), int(pos_y)
        x1, y1 = max(0, pos_x), max(0, pos_y)
        x2, y2 = min(canvas_w, pos_x + ov_w), min(canvas_h, pos_y + ov_h)
        if x2 <= x1 or y2 <= y1: return img
        ov_x1, ov_y1 = x1 - pos_x, y1 - pos_y
        ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)
        overlay_crop = img_overlay[ov_y1:ov_y2, ov_x1:ov_x2]
        overlay_rgb = overlay_crop[:, :, :3]
        mask_crop = alpha_mask[ov_y1:ov_y2, ov_x1:ov_x2] / 255.0
        mask_3ch = np.dstack([mask_crop] * 3)
        bg_crop = img[y1:y2, x1:x2]
        out_crop = (overlay_rgb * mask_3ch + bg_crop * (1.0 - mask_3ch)).astype(np.uint8)
        img[y1:y2, x1:x2] = out_crop
        return img

    def check_collision(self, new_rect, occupied_rects):
        nx, ny, nw, nh = new_rect
        for (ox, oy, ow, oh) in occupied_rects:
            if (nx < ox + ow and nx + nw > ox and ny < oy + oh and ny + nh > oy):
                return True
        return False

    def generate_pallet(self, specific_ptype=None):
        pallet = self.get_random_background()
        annotations = []
        if not self.loader.profiles: return pallet, [], {}

        if specific_ptype: ptype = specific_ptype
        else: ptype = random.choice(list(self.loader.profiles.keys()))

        profile_contour = self.loader.profiles[ptype]
        scale_factor = self.width / 640.0
        is_paired = ptype in SPECIAL_PAIRS_CONFIG
        
        pile_count = random.randint(1, 2)
        render_commands_depth = []
        render_commands_face = []
        occupied_rects = []

        for _ in range(pile_count):
            
            target_rotation = 0 # Varsayılan dik

            if is_paired:
                base_size = int(random.randint(60, 70) * scale_factor)
                
                # 1560 YATAYLIK MANTIĞI
                if ptype == '1560':
                    if random.random() < 0.5:
                        # --- YATAY DİZİLİM (BRICK MODE) ---
                        target_rotation = 90
                        # Yatay olunca genişlik > yükseklik olur.
                        # Üst üste dizmek için: Cols AZ, Rows ÇOK olmalı.
                        grid_cols = random.randint(1, 2)
                        grid_rows = random.randint(8, 14)
                    else:
                        # --- DİKEY DİZİLİM (NORMAL) ---
                        target_rotation = 0
                        grid_cols = random.randint(5, 12)
                        grid_rows = random.randint(1, 4)
                
                elif ptype == '1524':
                    if random.random() < 0.3:
                        grid_cols, grid_rows = 1, random.randint(5, 12) # Kule
                    else:
                        grid_cols, grid_rows = random.randint(3, 6), random.randint(3, 8) # Blok
                gap = 0 
            else:
                base_size = int(random.randint(20, 35) * scale_factor)
                grid_cols = random.randint(5, 15)
                grid_rows = random.randint(5, 15)
                gap = random.randint(2, 5)

            # Referans Sprite (Rotasyona dikkat et)
            ref_sprite, _ = self.renderer.render_profile_sprite(profile_contour, base_size, target_rotation, ptype)
            ph, pw = ref_sprite.shape[:2]
            if ph == 0 or pw == 0: continue
            
            pile_w = grid_cols * (pw + gap)
            pile_h = grid_rows * (ph + gap)
            
            max_x = self.width - pile_w
            max_y = self.height - pile_h
            if max_x < 0: max_x = 0
            if max_y < 0: max_y = 0

            found_pos = False
            for retry in range(50):
                start_x = random.randint(0, max_x)
                start_y = random.randint(0, max_y)
                new_rect = [start_x, start_y, pile_w, pile_h]
                if not self.check_collision(new_rect, occupied_rects):
                    occupied_rects.append(new_rect)
                    found_pos = True
                    break
            if not found_pos: continue
            
            for r in range(grid_rows):
                cur_y = start_y + r * (ph + gap)
                for c in range(grid_cols):
                    cur_x = start_x + c * (pw + gap)
                    if cur_x + pw > self.width or cur_y + ph > self.height: continue
                    
                    cur_sz = base_size if is_paired else int(base_size * random.uniform(0.95, 1.05))
                    
                    # Paired ise, önceden belirlediğimiz rotasyonu (0 veya 90) kullan
                    rot = target_rotation if is_paired else random.choice([0, 180])
                    
                    sprite, _ = self.renderer.render_profile_sprite(profile_contour, cur_sz, rot, ptype)
                    sh, sw = sprite.shape[:2]
                    
                    jx = random.randint(-1, 1) if is_paired else random.randint(-2, 2)
                    jy = random.randint(-1, 1) if is_paired else random.randint(-2, 2)
                    final_x = cur_x + jx
                    final_y = cur_y + jy
                    
                    center_x, center_y = self.width // 2, self.height // 2
                    obj_cx = final_x + sw // 2
                    obj_cy = final_y + sh // 2
                    vec_x = center_x - obj_cx
                    vec_y = center_y - obj_cy
                    dist = math.sqrt(vec_x**2 + vec_y**2)
                    max_dist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                    depth_len = (dist / max_dist) * (50 * scale_factor)
                    depth_len = min(depth_len, sw * 0.8)
                    
                    render_commands_depth.append({
                        'pos_x': final_x, 'pos_y': final_y,
                        'sw': sw, 'sh': sh,
                        'vec_x': vec_x, 'vec_y': vec_y, 'dist': dist,
                        'depth_len': depth_len,
                        'alpha': sprite[:,:,3]
                    })
                    render_commands_face.append({
                        'sprite': sprite,
                        'pos_x': final_x, 'pos_y': final_y,
                        'sw': sw, 'sh': sh,
                        'ptype_idx': list(self.loader.profiles.keys()).index(ptype)
                    })

        for cmd in render_commands_depth:
            if cmd['dist'] <= 0: continue
            norm_x = cmd['vec_x'] / cmd['dist']
            norm_y = cmd['vec_y'] / cmd['dist']
            total_offset_x = int(norm_x * cmd['depth_len'])
            total_offset_y = int(norm_y * cmd['depth_len'])
            body_color = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + 50
            steps = int(max(abs(total_offset_x), abs(total_offset_y))) 
            steps = min(steps, 30)
            for s in range(steps, 0, -1):
                ratio = s / steps if steps > 0 else 0
                ox = int(total_offset_x * ratio)
                oy = int(total_offset_y * ratio)
                self.overlay_image_alpha(pallet, body_color, cmd['pos_x']+ox, cmd['pos_y']+oy, cmd['alpha'])

        for cmd in render_commands_face:
            self.overlay_image_alpha(pallet, cmd['sprite'], cmd['pos_x'], cmd['pos_y'], cmd['sprite'][:,:,3])
            cx = (cmd['pos_x'] + cmd['sw']/2) / self.width
            cy = (cmd['pos_y'] + cmd['sh']/2) / self.height
            bw = cmd['sw'] / self.width
            bh = cmd['sh'] / self.height
            annotations.append({'class_id': cmd['ptype_idx'], 'bbox': [cx, cy, bw, bh]})

        return pallet, annotations, {}

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# --- 4. ANA ÇALIŞTIRMA ---
def main():
    if not os.path.exists(DXF_FOLDER): print("DXF yok."); return
    loader = DXFProfileLoader()
    dxf_files = glob.glob(os.path.join(DXF_FOLDER, "*_CLEAN.dxf"))
    if not dxf_files: dxf_files = glob.glob(os.path.join(DXF_FOLDER, "*.dxf"))
    for f in dxf_files:
        name = os.path.basename(f).split('_')[0].split('.')[0]
        loader.load_dxf(f, name)

    if not loader.profiles: print("Profil yok!"); return

    renderer = ProfileRenderer()
    generator = SyntheticPalletGenerator(loader, renderer, BG_FOLDER)
    
    if os.path.exists(BASE_DIR): shutil.rmtree(BASE_DIR)
    for d in ['images/train', 'labels/train', 'images/val', 'labels/val']: 
        os.makedirs(f'{BASE_DIR}/{d}', exist_ok=True)
        
    print(f"\n🚀 Sentetik Veri V14 (FIXED GEOMETRY) Başlıyor...")
    print(f"   - 1560 Dikey/Yatay Mantığı: Mükemmel Montaj + Sonradan Döndürme")
    print(f"   - Boşluk (Gap) Sorunu: ÇÖZÜLDÜ")
    print(f"   - Void Rengi: Gri Noise (Gerçekçi)")

    # --- CLASS BALANCING KUYRUĞU ---
    keys = list(loader.profiles.keys())
    num_classes = len(keys)
    if num_classes > 0:
        batch_size = IMAGE_COUNT // num_classes
        remainder = IMAGE_COUNT % num_classes
        production_queue = []
        for k in keys: production_queue.extend([k] * batch_size)
        production_queue.extend(random.choices(keys, k=remainder))
        random.shuffle(production_queue)
    else:
        production_queue = [None] * IMAGE_COUNT

    for i in range(IMAGE_COUNT):
        if i % 50 == 0: print(f"   -> {i}/{IMAGE_COUNT}")
        target_ptype = production_queue[i]
        pallet, anns, _ = generator.generate_pallet(specific_ptype=target_ptype)
        clahe_pallet = apply_clahe(pallet)
        subset = 'train' if i < IMAGE_COUNT * 0.9 else 'val'
        cv2.imwrite(f'{BASE_DIR}/images/{subset}/syn_{i}.jpg', pallet)
        cv2.imwrite(f'{BASE_DIR}/images/{subset}/syn_clahe_{i}.jpg', clahe_pallet)
        label_str = ""
        for a in anns: label_str += f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} {a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n"
        with open(f'{BASE_DIR}/labels/{subset}/syn_{i}.txt', 'w') as f: f.write(label_str)
        with open(f'{BASE_DIR}/labels/{subset}/syn_clahe_{i}.txt', 'w') as f: f.write(label_str)

    with open(f'{BASE_DIR}/data.yaml', 'w') as f:
        names = "\n".join([f"  {i}: {n}" for i, n in enumerate(loader.profiles.keys())])
        f.write(f"path: {os.path.abspath(BASE_DIR)}\ntrain: images/train\nval: images/val\nnames:\n{names}\nnc: {len(loader.profiles)}")

    print(f"\n✅ V14 TAMAMLANDI! (Klasör: {BASE_DIR})")

if __name__ == "__main__":
    main()