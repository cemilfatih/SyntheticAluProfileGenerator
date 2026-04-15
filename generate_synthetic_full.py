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
IMAGES_PER_CLASS_TRAIN = 5  
IMAGES_PER_CLASS_VAL = 1      
IMG_WIDTH = 1280
IMG_HEIGHT = 1280
BASE_DIR = "datasets/deneme" 
BG_FOLDER = "background"
DXF_FOLDER = "cad_files"
FALSE_SAMPLES_DIR = "false_samples"

# Özel Profil Ayarları
SPECIAL_PAIRS_CONFIG = {
    '1524': 0.70,  # Kare
    '1560': 0.50   # Dikdörtgen
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
        except: return None

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
        is_shiny = random.random() < 0.30
        if is_shiny:
            base_val = random.randint(230, 255); contrast_val = 40
        else:
            base_val = random.randint(180, 220); contrast_val = 25

        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        if random.random() < 0.5:
            for y in range(h):
                v = int(base_val - contrast_val * np.sin(y / h * 3.14))
                img_bgr[y, :, :] = (v, v, v)
        else:
            for x in range(w):
                v = int(base_val - contrast_val * np.sin(x / w * 3.14))
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
        thickness = int(min(w, h) * 0.15) 
        return self.apply_aluminum_texture(cropped_mask), thickness

    def create_paired_block(self, contour, size, profile_name, target_rotation=0):
        sprite_a, t = self.get_single_half(contour, size, 0)
        if sprite_a is None: return np.zeros((10,10,4), dtype=np.uint8), (0,0,0,0), 0
        h, w = sprite_a.shape[:2]
        sprite_b_raw, _ = self.get_single_half(contour, size, 180)
        sprite_b = cv2.resize(sprite_b_raw, (w, h))
        
        tightness = SPECIAL_PAIRS_CONFIG.get(profile_name, 1.0)
        shift_val = int(t * tightness)
        
        canvas_w = w * 2 + shift_val * 4
        canvas_h = h * 2 + shift_val * 4
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        cx, cy = canvas_w // 2, canvas_h // 2
        
        pos_a_x = cx - w // 2; pos_a_y = cy - h // 2
        pos_b_x = pos_a_x + shift_val; pos_b_y = pos_a_y + shift_val
        
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
            return canvas[y:y+h_crop, x:x+w_crop], (x, y, w_crop, h_crop), t
        return canvas, (0,0,0,0), 0

    def render_profile_sprite(self, contour, size, rotation=0, profile_name="default"):
        if profile_name in SPECIAL_PAIRS_CONFIG:
            return self.create_paired_block(contour, size, profile_name, target_rotation=rotation)
        
        sprite, t = self.get_single_half(contour, size, rotation)
        if sprite is None: return np.zeros((10,10,4), dtype=np.uint8), (0,0,0,0), 0
        return sprite, (0,0,sprite.shape[1], sprite.shape[0]), t

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

    # --- PERSPEKTİF TÜNEL EFEKTİ (PITCH BLACK MOD) ---
    def generate_tunnel_void(self, w, h, obj_cx, obj_cy, thickness, ptype):
        screen_cx, screen_cy = self.width // 2, self.height // 2
        void_w = w - 2 * thickness; void_h = h - 2 * thickness
        if void_w <= 0 or void_h <= 0: return None
        
        vec_x = screen_cx - obj_cx; vec_y = screen_cy - obj_cy
        light_center_x = np.clip((void_w // 2) + int(vec_x * 0.3), 0, void_w)
        light_center_y = np.clip((void_h // 2) + int(vec_y * 0.3), 0, void_h)
        
        Y, X = np.ogrid[:void_h, :void_w]
        dist_map = np.sqrt((X - light_center_x)**2 + (Y - light_center_y)**2)
        max_dist = np.sqrt(void_w**2 + void_h**2)
        norm_dist = np.clip(dist_map / (max_dist * 0.6), 0, 1)
        
        # --- V24: ZİFİRİ KARANLIK OPAKIK AYARLARI ---
        if ptype == '1524':
            # 1524: %92 - %99 (Neredeyse Simsiyah)
            base_opacity = random.uniform(0.92, 0.99)
        else:
            # 1560: %80 - %95 (Çok Koyu)
            base_opacity = random.uniform(0.80, 0.95)
            
        alpha_channel = (np.power(norm_dist, 0.7) * base_opacity * 255).astype(np.uint8)
        
        base_color = np.zeros((void_h, void_w, 3), dtype=np.uint8)
        noise = np.random.randint(0, 20, (void_h, void_w, 3)) # Noise daha az
        base_color = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        
        return cv2.merge([base_color[:,:,0], base_color[:,:,1], base_color[:,:,2], alpha_channel])

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
            if (nx < ox + ow and nx + nw > ox and ny < oy + oh and ny + nh > oy): return True
        return False

    def generate_pallet(self, specific_ptype):
        pallet = self.get_random_background()
        annotations = []
        scale_factor = self.width / 640.0
        
        if specific_ptype in SPECIAL_PAIRS_CONFIG:
            pile_count = random.randint(1, 2)
            render_commands_depth = []
            render_commands_face = []
            occupied_rects = []

            for _ in range(pile_count):
                current_pile_ptype = specific_ptype
                if random.random() < 0.50:
                    other_specials = [s for s in SPECIAL_PAIRS_CONFIG.keys() if s != specific_ptype]
                    if other_specials: current_pile_ptype = random.choice(other_specials)
                
                profile_contour = self.loader.profiles[current_pile_ptype]
                
                if current_pile_ptype == '1560': base_size = int(random.randint(100, 120) * scale_factor)
                else: base_size = int(random.randint(50, 60) * scale_factor)

                target_rotation = 0; gap = 0
                if current_pile_ptype == '1560':
                    if random.random() < 0.5:
                        target_rotation = 90; grid_cols, grid_rows = random.randint(1, 2), random.randint(8, 14)
                    else:
                        target_rotation = 0; grid_cols, grid_rows = random.randint(5, 12), random.randint(1, 4)
                elif current_pile_ptype == '1524':
                    shape_roll = random.random()
                    if shape_roll < 0.3: grid_cols, grid_rows = 1, random.randint(5, 12)
                    elif shape_roll < 0.6: grid_cols, grid_rows = random.randint(4, 10), 1
                    else: grid_cols, grid_rows = random.randint(3, 6), random.randint(3, 8)
                
                ref_sprite, _, thickness = self.renderer.render_profile_sprite(profile_contour, base_size, target_rotation, current_pile_ptype)
                ph, pw = ref_sprite.shape[:2]
                if ph == 0 or pw == 0: continue
                
                pile_w = grid_cols * (pw + gap)
                pile_h = grid_rows * (ph + gap)
                max_x, max_y = self.width - pile_w, self.height - pile_h
                if max_x < 0: max_x = 0; 
                if max_y < 0: max_y = 0

                found_pos = False
                for retry in range(50):
                    start_x, start_y = random.randint(0, max_x), random.randint(0, max_y)
                    new_rect = [start_x, start_y, pile_w, pile_h]
                    if not self.check_collision(new_rect, occupied_rects):
                        occupied_rects.append(new_rect); found_pos = True; break
                if not found_pos: continue

                for r in range(grid_rows):
                    cur_y = start_y + r * (ph + gap)
                    for c in range(grid_cols):
                        cur_x = start_x + c * (pw + gap)
                        if cur_x + pw > self.width or cur_y + ph > self.height: continue
                        
                        rot = target_rotation
                        sprite, _, t = self.renderer.render_profile_sprite(profile_contour, base_size, rot, current_pile_ptype)
                        sh, sw = sprite.shape[:2]
                        
                        jx, jy = random.randint(-1, 1), random.randint(-1, 1)
                        final_x, final_y = cur_x + jx, cur_y + jy
                        
                        cx, cy = self.width // 2, self.height // 2
                        obj_cx, obj_cy = final_x + sw // 2, final_y + sh // 2
                        vec_x, vec_y = cx - obj_cx, cy - obj_cy
                        dist = math.sqrt(vec_x**2 + vec_y**2)
                        max_dist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                        depth_len = min((dist/max_dist)*(50*scale_factor), sw*0.8)
                        
                        # Tünel Gölgesi (PITCH BLACK MOD)
                        void_sprite = self.generate_tunnel_void(sw, sh, obj_cx, obj_cy, t, current_pile_ptype)
                        if void_sprite is not None:
                            self.overlay_image_alpha(pallet, void_sprite, final_x+t, final_y+t, void_sprite[:,:,3])

                        render_commands_depth.append({
                            'pos_x': final_x, 'pos_y': final_y, 'sw': sw, 'sh': sh,
                            'vec_x': vec_x, 'vec_y': vec_y, 'dist': dist, 'depth_len': depth_len,
                            'alpha': sprite[:,:,3]
                        })
                        render_commands_face.append({
                            'sprite': sprite, 'pos_x': final_x, 'pos_y': final_y, 'sw': sw, 'sh': sh,
                            'ptype_idx': list(self.loader.profiles.keys()).index(current_pile_ptype)
                        })

            for cmd in render_commands_depth:
                if cmd['dist'] <= 0: continue
                nx, ny = cmd['vec_x']/cmd['dist'], cmd['vec_y']/cmd['dist']
                tox, toy = int(nx*cmd['depth_len']), int(ny*cmd['depth_len'])
                body_color = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + 50
                steps = min(int(max(abs(tox), abs(toy))), 30)
                for s in range(steps, 0, -1):
                    r = s/steps if steps>0 else 0
                    self.overlay_image_alpha(pallet, body_color, cmd['pos_x']+int(tox*r), cmd['pos_y']+int(toy*r), cmd['alpha'])
            
            for cmd in render_commands_face:
                self.overlay_image_alpha(pallet, cmd['sprite'], cmd['pos_x'], cmd['pos_y'], cmd['sprite'][:,:,3])
                bx, by = (cmd['pos_x']+cmd['sw']/2)/self.width, (cmd['pos_y']+cmd['sh']/2)/self.height
                annotations.append({'class_id': cmd['ptype_idx'], 'bbox': [bx, by, cmd['sw']/self.width, cmd['sh']/self.height]})

        else:
            # === ESKİ MANTIK ===
            profile_contour = self.loader.profiles[specific_ptype]
            base_size = int(random.randint(25, 40) * scale_factor)
            spacer_range = (int(5*scale_factor), int(15*scale_factor))
            gap_range = (int(2*scale_factor), int(5*scale_factor))
            
            ref_sprite, _, _ = self.renderer.render_profile_sprite(profile_contour, base_size, 0, specific_ptype)
            ph, pw = ref_sprite.shape[:2]
            if ph == 0 or pw == 0: return pallet, [], {}
            
            current_y = self.height - int(10 * scale_factor)
            render_commands_depth = []
            render_commands_face = []

            while current_y > int(50 * scale_factor):
                spacer = random.randint(*spacer_range)
                row_y = current_y - spacer - ph
                if row_y < 0: break
                
                current_x = random.randint(0, int(30*scale_factor))
                while current_x < self.width - pw:
                    cur_rot = random.choice([0, 180])
                    cur_sz = int(base_size * random.uniform(0.95, 1.05))
                    sprite, _, _ = self.renderer.render_profile_sprite(profile_contour, cur_sz, cur_rot, specific_ptype)
                    sh, sw = sprite.shape[:2]
                    pos_x = current_x; pos_y = row_y + random.randint(-2, 2)
                    if pos_x + sw > self.width: break
                    
                    cx, cy = self.width // 2, self.height // 2
                    obj_cx, obj_cy = pos_x + sw // 2, pos_y + sh // 2
                    vec_x, vec_y = cx - obj_cx, cy - obj_cy
                    dist = math.sqrt(vec_x**2 + vec_y**2)
                    max_dist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                    depth_len = min((dist/max_dist)*(50*scale_factor), sw*0.8)

                    render_commands_depth.append({
                        'pos_x': pos_x, 'pos_y': pos_y, 'sw': sw, 'sh': sh,
                        'vec_x': vec_x, 'vec_y': vec_y, 'dist': dist, 'depth_len': depth_len,
                        'alpha': sprite[:,:,3]
                    })
                    render_commands_face.append({
                        'sprite': sprite, 'pos_x': pos_x, 'pos_y': pos_y, 'sw': sw, 'sh': sh
                    })
                    gap = random.randint(*gap_range)
                    current_x += (sw + gap)
                current_y = row_y

            for cmd in render_commands_depth:
                if cmd['dist'] <= 0: continue
                nx, ny = cmd['vec_x']/cmd['dist'], cmd['vec_y']/cmd['dist']
                tox, toy = int(nx*cmd['depth_len']), int(ny*cmd['depth_len'])
                body_color = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + 50
                steps = min(int(max(abs(tox), abs(toy))), 30)
                for s in range(steps, 0, -1):
                    r = s/steps if steps>0 else 0
                    self.overlay_image_alpha(pallet, body_color, cmd['pos_x']+int(tox*r), cmd['pos_y']+int(toy*r), cmd['alpha'])
            
            for cmd in render_commands_face:
                self.overlay_image_alpha(pallet, cmd['sprite'], cmd['pos_x'], cmd['pos_y'], cmd['sprite'][:,:,3])
                bx, by = (cmd['pos_x']+cmd['sw']/2)/self.width, (cmd['pos_y']+cmd['sh']/2)/self.height
                annotations.append({'class_id': list(self.loader.profiles.keys()).index(specific_ptype), 'bbox': [bx, by, cmd['sw']/self.width, cmd['sh']/self.height]})

        if random.random() < 0.3:
            noise = np.random.normal(0, 3, pallet.shape).astype(np.int16)
            pallet = np.clip(pallet.astype(np.int16) + noise, 0, 255).astype(np.uint8)

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
    dxf_files = glob.glob(os.path.join(DXF_FOLDER, "*.dxf"))
    for f in dxf_files:
        name = os.path.basename(f).split('_')[0].split('.')[0]
        loader.load_dxf(f, name)

    if not loader.profiles: print("Profil yok!"); return

    renderer = ProfileRenderer()
    generator = SyntheticPalletGenerator(loader, renderer, BG_FOLDER)
    
    if os.path.exists(BASE_DIR): shutil.rmtree(BASE_DIR)
    for subset in ['train', 'val']:
        os.makedirs(f'{BASE_DIR}/images/{subset}', exist_ok=True)
        os.makedirs(f'{BASE_DIR}/labels/{subset}', exist_ok=True)
        
    print(f"\n🚀 Sentetik Veri V24 (PITCH BLACK TUNNEL) Başlıyor...")
    
    class_names = list(loader.profiles.keys())
    
    for ptype in class_names:
        print(f"\n📦 İşleniyor: {ptype} ...")
        
        for i in range(IMAGES_PER_CLASS_TRAIN):
            if i % 100 == 0: print(f"   [Train] {i}/{IMAGES_PER_CLASS_TRAIN}")
            pallet, anns, _ = generator.generate_pallet(specific_ptype=ptype)
            name = f"{ptype}_train_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/train/{name}.jpg', pallet)
            label_str = ""
            for a in anns: label_str += f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} {a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n"
            with open(f'{BASE_DIR}/labels/train/{name}.txt', 'w') as f: f.write(label_str)

        for i in range(IMAGES_PER_CLASS_VAL):
            if i % 10 == 0: print(f"   [Val] {i}/{IMAGES_PER_CLASS_VAL}")
            pallet, anns, _ = generator.generate_pallet(specific_ptype=ptype)
            name = f"{ptype}_val_{i}"
            cv2.imwrite(f'{BASE_DIR}/images/val/{name}.jpg', pallet)
            label_str = ""
            for a in anns: label_str += f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} {a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n"
            with open(f'{BASE_DIR}/labels/val/{name}.txt', 'w') as f: f.write(label_str)

    with open(f'{BASE_DIR}/data.yaml', 'w') as f:
        names = "\n".join([f"  {i}: {n}" for i, n in enumerate(class_names)])
        f.write(f"path: {os.path.abspath(BASE_DIR)}\ntrain: images/train\nval: images/val\nnames:\n{names}\nnc: {len(class_names)}")

    print(f"\n✅ V24 TAMAMLANDI! (Klasör: {BASE_DIR})")

if __name__ == "__main__":
    main()