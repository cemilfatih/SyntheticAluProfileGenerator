import ezdxf
import cv2
import numpy as np
import random
import os
import math
import glob
import shutil

# --- AYARLAR ---
IMAGE_COUNT = 5000          
IMG_WIDTH = 1280            
IMG_HEIGHT = 1280           
BASE_DIR = "datasets/dataset_final_v7_9_layered" 
BG_FOLDER = "background"
DXF_FOLDER = "cad_files"
FALSE_SAMPLES_DIR = "false_samples"

# --- 1. DXF YÜKLEYİCİ ---
class DXFProfileLoader:
    def __init__(self): self.profiles = {}
    def load_dxf(self, dxf_path, profile_name):
        try: doc = ezdxf.readfile(dxf_path); msp = doc.modelspace()
        except: return None
        all_points = []
        for entity in msp:
            if entity.dxftype() == 'LINE': all_points.extend([[entity.dxf.start.x, entity.dxf.start.y], [entity.dxf.end.x, entity.dxf.end.y]])
            elif entity.dxftype() == 'LWPOLYLINE': all_points.extend(list(entity.get_points('xy')))
            elif entity.dxftype() == 'POLYLINE': 
                 for v in entity.vertices: all_points.append([v.dxf.location.x, v.dxf.location.y])
            elif entity.dxftype() == 'ARC': all_points.extend(self.arc_to_points(entity))
            elif entity.dxftype() == 'CIRCLE': all_points.extend(self.circle_to_points(entity))
        if all_points: contour = self.create_closed_contour(all_points); self.profiles[profile_name] = contour; return contour
        return None
    def arc_to_points(self, entity, s=30):
        c, r, start, end = entity.dxf.center, entity.dxf.radius, math.radians(entity.dxf.start_angle), math.radians(entity.dxf.end_angle)
        if end < start: end += 2 * math.pi
        return [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(start, end, s)]
    def circle_to_points(self, entity, s=60):
        c, r = entity.dxf.center, entity.dxf.radius
        return [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, s)]
    def create_closed_contour(self, points):
        pts = np.array(points); pts = pts - pts.mean(axis=0); max_val = np.max(np.abs(pts))
        return pts / max_val if max_val > 0 else pts

# --- 2. RENDER MOTORU ---
class ProfileRenderer:
    def __init__(self): pass
    
    def add_scratches(self, img, mask):
        if random.random() > 0.6: return img 
        h, w = img.shape[:2]
        scratch_layer = np.zeros_like(img)
        count = random.randint(1, 5)
        for _ in range(count):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            thickness = random.randint(1, 2)
            cv2.line(scratch_layer, pt1, pt2, color, thickness)
        return cv2.add(img, cv2.bitwise_and(scratch_layer, scratch_layer, mask=mask))

    def render_profile_sprite(self, contour, size, rotation=0):
        canvas_size = int(size * 4.0)
        img_bgra = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        scaled_contour = contour * size + canvas_size // 2
        
        if rotation != 0:
            M = cv2.getRotationMatrix2D((canvas_size//2, canvas_size//2), rotation, 1.0)
            ones = np.ones((len(scaled_contour), 1))
            scaled_contour = M.dot(np.hstack([scaled_contour, ones]).T).T
        
        pts = scaled_contour.astype(np.int32)
        
        base_val = random.randint(110, 190)
        contrast = random.randint(30, 60)
        light_val = min(255, base_val + contrast)
        dark_val = max(50, base_val - contrast)
        
        gradient_bgr = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        direction = random.choice(['v', 'h'])
        if direction == 'v':
            for y in range(canvas_size):
                v = int(dark_val + (light_val - dark_val) * (y/canvas_size))
                gradient_bgr[y, :, :] = (v, v, v)
        else:
            for x in range(canvas_size):
                v = int(dark_val + (light_val - dark_val) * (x/canvas_size))
                gradient_bgr[:, x, :] = (v, v, v)
        
        noise = np.random.randint(-10, 10, gradient_bgr.shape)
        gradient_bgr = np.clip(gradient_bgr.astype(int) + noise, 0, 255).astype(np.uint8)
        
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        if size > 15:
            mask = cv2.GaussianBlur(mask, (5,5), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        profile_bgr = cv2.bitwise_and(gradient_bgr, gradient_bgr, mask=mask)
        profile_bgr = self.add_scratches(profile_bgr, mask)
        
        cv2.polylines(profile_bgr, [pts], True, (40, 40, 40), 1, cv2.LINE_AA)
        
        b, g, r = cv2.split(profile_bgr)
        img_bgra = cv2.merge((b, g, r, mask))
        
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            pad = 4
            x, y = max(0, x-pad), max(0, y-pad)
            w, h = min(canvas_size-x, w+2*pad), min(canvas_size-y, h+2*pad)
            return img_bgra[y:y+h, x:x+w], (x, y, w, h)
        return img_bgra, (0,0,0,0)

# --- 3. PALET ÜRETİCİ ---
class SyntheticPalletGenerator:
    def __init__(self, profile_loader, renderer, bg_folder='background'):
        self.loader = profile_loader
        self.renderer = renderer
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.bg_images = glob.glob(os.path.join(bg_folder, "*.*"))

    def get_random_background(self):
        is_dark_mode = random.random() < 0.50
        if self.bg_images:
            bg_path = random.choice(self.bg_images)
            bg = cv2.imread(bg_path)
            if bg is not None:
                bg = cv2.resize(bg, (self.width, self.height))
                factor = random.uniform(0.15, 0.40) if is_dark_mode else random.uniform(0.5, 0.8)
                bg = (bg.astype(float) * factor).astype(np.uint8)
                return bg
        val = random.randint(20, 50) if is_dark_mode else random.randint(50, 100)
        bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * val
        noise = np.random.randint(-5, 5, bg.shape)
        return np.clip(bg + noise, 0, 255).astype(np.uint8)

    def add_rack_posts(self, img):
        if random.random() > 0.60: return img 
        h, w = img.shape[:2]
        scale_ratio = self.width / 640.0
        for side in ['left', 'right']:
            if random.random() < 0.6:
                bar_w = random.randint(int(30*scale_ratio), int(90*scale_ratio))
                color = np.array([30, 40, 50])
                start_x = 0 if side == 'left' else w - bar_w
                cv2.rectangle(img, (start_x, 0), (start_x + bar_w, h), color.tolist(), -1)
                noise = np.random.randint(-20, 20, (h, bar_w, 3))
                roi = img[0:h, start_x:start_x+bar_w]
                roi[:] = np.clip(roi.astype(int) + noise, 0, 255).astype(np.uint8)
        return img

    def overlay_image_alpha(self, img, img_overlay, pos_x, pos_y, alpha_mask):
        h, w = img_overlay.shape[:2]
        if pos_x >= img.shape[1] or pos_y >= img.shape[0]: return img
        if pos_x < 0 or pos_y < 0: return img 
        if pos_x + w > img.shape[1]: w = img.shape[1] - pos_x
        if pos_y + h > img.shape[0]: h = img.shape[0] - pos_y
        if w <= 0 or h <= 0: return img

        overlay_crop = img_overlay[0:h, 0:w]
        bg_crop = img[pos_y:pos_y+h, pos_x:pos_x+w]
        mask_crop = alpha_mask[0:h, 0:w] / 255.0
        mask_3ch = np.dstack([mask_crop] * 3)
        out_crop = (overlay_crop * mask_3ch + bg_crop * (1.0 - mask_3ch)).astype(np.uint8)
        img[pos_y:pos_y+h, pos_x:pos_x+w] = out_crop
        return img

    def generate_pallet(self):
        pallet = self.get_random_background()
        annotations = []
        if not self.loader.profiles: return pallet, [], {}

        ptype = random.choice(list(self.loader.profiles.keys()))
        profile_contour = self.loader.profiles[ptype]
        scale_factor = self.width / 640.0
        
        density_mode = random.choices(['sparse', 'medium', 'dense', 'packed'], [0.1, 0.2, 0.35, 0.35])[0]
        
        if density_mode == 'sparse': 
            base_size = int(random.randint(35, 50) * scale_factor); spacer_range = (int(30*scale_factor), int(50*scale_factor)); fill_limit_y = int(300 * scale_factor)
        elif density_mode == 'medium': 
            base_size = int(random.randint(20, 30) * scale_factor); spacer_range = (int(10*scale_factor), int(20*scale_factor)); fill_limit_y = int(100 * scale_factor)
        else: 
            base_size = int(random.randint(10, 15) * scale_factor); spacer_range = (0, int(3*scale_factor)); fill_limit_y = int(10 * scale_factor)

        is_nesting_type = '7170' in ptype
        base_rotation_angle = 90 if is_nesting_type else 0
        if not is_nesting_type:
             test_img, _ = self.renderer.render_profile_sprite(profile_contour, base_size, 0)
             if test_img.shape[0] > test_img.shape[1]: base_rotation_angle = 90

        ref_img, _ = self.renderer.render_profile_sprite(profile_contour, base_size, base_rotation_angle)
        ph, pw = ref_img.shape[:2]
        if ph == 0 or pw == 0: return pallet, [], {}

        current_y = self.height - int(5 * scale_factor)
        center_x, center_y = self.width // 2, self.height // 2
        
        # --- MİMARİ DEĞİŞİKLİK: ERTELENMİŞ ÇİZİM LİSTESİ ---
        # Ön yüzleri hemen çizmiyoruz, bu listeye atıyoruz.
        deferred_faces = []

        while current_y > fill_limit_y:
            spacer = random.randint(*spacer_range)
            row_y = current_y - spacer - ph
            if row_y < 0: break
            current_x = random.randint(0, int(30*scale_factor))
            row_flip = random.choice([0, 180]) 
            is_tight_row = random.random() < 0.50
            
            while current_x < self.width - pw:
                cur_size = int(base_size * random.uniform(0.98, 1.02))
                jitter = random.uniform(-1, 1)
                rot = base_rotation_angle + row_flip + jitter if is_nesting_type else base_rotation_angle + random.choice([0, 180]) + jitter
                
                sprite, _ = self.renderer.render_profile_sprite(profile_contour, cur_size, rot)
                sh, sw = sprite.shape[:2]
                
                pos_x, pos_y = current_x, row_y + random.randint(-1, 1)
                if pos_x + sw > self.width: break
                
                if pos_y >= 0 and pos_x >= 0 and pos_y+sh <= self.height and pos_x+sw <= self.width:
                    
                    # --- FAZ 1: SADECE DERİNLİK GÖVDESİNİ ÇİZ ---
                    obj_cx, obj_cy = pos_x + sw // 2, pos_y + sh // 2
                    vec_x, vec_y = center_x - obj_cx, center_y - obj_cy
                    dist = math.sqrt(vec_x**2 + vec_y**2)
                    max_dist = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                    
                    raw_depth = (dist / max_dist) * (40 * scale_factor)
                    max_allowed_depth = cur_size * 0.6 
                    depth_factor = min(raw_depth, max_allowed_depth)
                    
                    if dist > 0:
                        norm_x, norm_y = vec_x / dist, vec_y / dist
                        alpha = sprite[:, :, 3]
                        body_val = random.randint(50, 80)
                        body_color = np.zeros((sh, sw, 3), dtype=np.uint8) + body_val
                        
                        total_offset_x = int(norm_x * depth_factor)
                        total_offset_y = int(norm_y * depth_factor)
                        
                        steps = int(max(abs(total_offset_x), abs(total_offset_y)) * 2) + 3
                        steps = min(steps, 50)
                        
                        for s in range(steps, 0, -1): 
                            ratio = s / steps
                            offset_x = int(total_offset_x * ratio)
                            offset_y = int(total_offset_y * ratio)
                            # Derinliği hemen palete basıyoruz
                            self.overlay_image_alpha(pallet, body_color, pos_x + offset_x, pos_y + offset_y, alpha)

                    # --- ÖN YÜZÜ ÇİZME! LİSTEYE EKLE. ---
                    # Gerekli tüm bilgileri sakla.
                    deferred_faces.append({
                        'sprite': sprite,
                        'pos_x': pos_x,
                        'pos_y': pos_y,
                        'sw': sw, 'sh': sh,
                        'ptype_idx': list(self.loader.profiles.keys()).index(ptype)
                    })
                
                if is_nesting_type:
                    overlap = int(sw * random.uniform(0.15, 0.25))
                    current_x += (sw - overlap)
                else:
                    gap = random.randint(int(-2 * scale_factor), int(1 * scale_factor)) if is_tight_row else random.randint(int(2 * scale_factor), int(6 * scale_factor))
                    current_x += (sw + gap)
            current_y = row_y 

        # --- FAZ 2: TÜM ÖN YÜZLERİ EN ÜSTE ÇİZ ---
        # Döngü bitti, tüm derinlikler çizildi. Şimdi ön yüzleri basıyoruz.
        for item in deferred_faces:
            sprite = item['sprite']
            pos_x, pos_y = item['pos_x'], item['pos_y']
            sw, sh = item['sw'], item['sh']
            
            # Gölge
            alpha = sprite[:, :, 3]
            rgb = sprite[:, :, :3]
            shadow_mask = cv2.GaussianBlur(alpha, (5,5), 0)
            shadow_offset = int(3 * scale_factor) 
            shadow_x, shadow_y = pos_x + shadow_offset, pos_y + shadow_offset
            
            if shadow_x+sw < self.width and shadow_y+sh < self.height:
                roi_shadow = pallet[shadow_y:shadow_y+sh, shadow_x:shadow_x+sw]
                shadow_factor = (shadow_mask / 255.0) * 0.4 
                shadow_factor = np.dstack([shadow_factor]*3)
                roi_shadow[:] = (roi_shadow * (1.0 - shadow_factor)).astype(np.uint8)

            # Ön Yüzü Yapıştır (En üst katman garanti)
            self.overlay_image_alpha(pallet, rgb, pos_x, pos_y, alpha)
            
            # Annotation ekle
            cx, cy = (pos_x + sw/2) / self.width, (pos_y + sh/2) / self.height
            annotations.append({'class_id': item['ptype_idx'], 'bbox': [cx, cy, sw/float(self.width), sh/float(self.height)]})

        pallet = self.add_rack_posts(pallet)
        
        if random.random() < 0.5:
            vignette_mask = np.zeros((self.height, self.width, 3), dtype=np.float32)
            radius = int(max(self.width, self.height) * random.uniform(0.7, 1.0))
            center = (self.width // 2, self.height // 2)
            cv2.circle(vignette_mask, center, radius, (1, 1, 1), -1, cv2.LINE_AA)
            blur_ksize = (int(self.width//5)|1, int(self.height//5)|1)
            vignette_mask = cv2.GaussianBlur(vignette_mask, blur_ksize, 0)
            darkness_intensity = random.uniform(0.5, 0.8)
            vignette_mask = vignette_mask * darkness_intensity + (1.0 - darkness_intensity)
            pallet = (pallet.astype(np.float32) * vignette_mask).clip(0, 255).astype(np.uint8)

        if random.random() < 0.8:
            noise_sigma = random.randint(2, 3) 
            noise = np.random.normal(0, noise_sigma, pallet.shape).astype(np.int16)
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
    for f in glob.glob(os.path.join(DXF_FOLDER, "*.dxf")): loader.load_dxf(f, os.path.splitext(os.path.basename(f))[0])
    if not loader.profiles: print("Profil yok!"); return

    renderer = ProfileRenderer()
    generator = SyntheticPalletGenerator(loader, renderer, BG_FOLDER)
    
    if os.path.exists(BASE_DIR): shutil.rmtree(BASE_DIR)
    for d in ['images/train', 'labels/train', 'images/val', 'labels/val']: os.makedirs(f'{BASE_DIR}/{d}', exist_ok=True)
        
    print(f"\n🚀 Sentetik Veri V7.9 (GUARANTEED LAYERING) Başlıyor...")
    print(f"Hedef: {IMAGE_COUNT} adet. Derinlikler asla ön yüzü kapatamaz.")

    for i in range(IMAGE_COUNT):
        if i % 100 == 0: print(f"İlerleme: {i}/{IMAGE_COUNT}")
        pallet, anns, _ = generator.generate_pallet()
        clahe_pallet = apply_clahe(pallet)
        
        cv2.imwrite(f'{BASE_DIR}/images/train/syn_{i}.jpg', pallet)
        cv2.imwrite(f'{BASE_DIR}/images/train/syn_clahe_{i}.jpg', clahe_pallet)
        
        label_str = ""
        for a in anns: label_str += f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} {a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n"
        with open(f'{BASE_DIR}/labels/train/syn_{i}.txt', 'w') as f: f.write(label_str)
        with open(f'{BASE_DIR}/labels/train/syn_clahe_{i}.txt', 'w') as f: f.write(label_str)

    if os.path.exists(FALSE_SAMPLES_DIR):
        print("\nNegatif Örnekler ekleniyor...")
        for idx, fpath in enumerate(glob.glob(os.path.join(FALSE_SAMPLES_DIR, "*.*"))):
            img = cv2.imread(fpath)
            if img is None: continue
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            basename = f"neg_{idx}"
            cv2.imwrite(f'{BASE_DIR}/images/train/{basename}.jpg', img)
            open(f'{BASE_DIR}/labels/train/{basename}.txt', 'w').close()

    with open(f'{BASE_DIR}/data.yaml', 'w') as f:
        names = "\n".join([f"  {i}: {n}" for i, n in enumerate(loader.profiles.keys())])
        f.write(f"path: {os.path.abspath(BASE_DIR)}\ntrain: images/train\nval: images/train\nnames:\n{names}\nnc: {len(loader.profiles)}")

    print(f"\n✅ V7.9 TAMAMLANDI!")

if __name__ == "__main__":
    main()