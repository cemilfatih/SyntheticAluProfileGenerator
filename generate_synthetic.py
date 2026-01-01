import ezdxf
import cv2
import numpy as np
import random
import os
import math
import glob

IMAGE_COUNT = 20

# --- 1. DXF YÜKLEYİCİ ---
class DXFProfileLoader:
    def __init__(self):
        self.profiles = {}
    
    def load_dxf(self, dxf_path, profile_name):
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
        except Exception as e:
            print(f"Hata: {dxf_path} okunamadı. {e}")
            return None
        
        all_points = []
        for entity in msp:
            if entity.dxftype() == 'LINE':
                all_points.extend([[entity.dxf.start.x, entity.dxf.start.y], [entity.dxf.end.x, entity.dxf.end.y]])
            elif entity.dxftype() == 'LWPOLYLINE':
                all_points.extend(list(entity.get_points('xy')))
            elif entity.dxftype() == 'POLYLINE':
                for vertex in entity.vertices:
                    all_points.append([vertex.dxf.location.x, vertex.dxf.location.y])
            elif entity.dxftype() == 'ARC':
                all_points.extend(self.arc_to_points(entity))
            elif entity.dxftype() == 'CIRCLE':
                all_points.extend(self.circle_to_points(entity))
        
        if all_points:
            contour = self.create_closed_contour(all_points)
            self.profiles[profile_name] = contour
            return contour
        return None

    def arc_to_points(self, entity, segments=30):
        center = entity.dxf.center
        radius = entity.dxf.radius
        start = math.radians(entity.dxf.start_angle)
        end = math.radians(entity.dxf.end_angle)
        if end < start: end += 2 * math.pi
        return [[center.x + radius * math.cos(a), center.y + radius * math.sin(a)] for a in np.linspace(start, end, segments)]

    def circle_to_points(self, entity, segments=60):
        c, r = entity.dxf.center, entity.dxf.radius
        return [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, segments)]
    
    def create_closed_contour(self, points):
        pts = np.array(points)
        pts = pts - pts.mean(axis=0)
        return pts / np.max(np.abs(pts)) if np.max(np.abs(pts)) > 0 else pts

# --- 2. RENDER MOTORU (V10: DENGELİ VERSİYON) ---
class ProfileRenderer:
    def __init__(self):
        pass
    
    def render_profile(self, contour, size, rotation=0, color_tint='neutral'):
        # Canvası geniş tut
        canvas_size = int(size * 3.0)
        img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        scaled_contour = contour * size + canvas_size // 2
        if rotation != 0:
            M = cv2.getRotationMatrix2D((canvas_size//2, canvas_size//2), rotation, 1.0)
            ones = np.ones((len(scaled_contour), 1))
            scaled_contour = M.dot(np.hstack([scaled_contour, ones]).T).T
        
        pts = scaled_contour.astype(np.int32)
        
        # --- DÜZELTME 1: RENK AYARI (Çok beyaz olmasın) ---
        # 255'e dayayınca detay kayboluyor. 160-220 arası ideal metal grisidir.
        base_gray = random.randint(160, 220) 
        
        if color_tint == 'yellow': 
            color = (base_gray - 30, base_gray, base_gray + 10)
        elif color_tint == 'blue':
            color = (base_gray + 10, base_gray, base_gray - 10)
        else:
            color = (base_gray, base_gray, base_gray)
            
        # --- DÜZELTME 2: GÜÇLÜ DOKU (High Contrast Noise) ---
        # Doku belli olsun diye gürültü aralığını artırdık (-20, 20)
        noise = np.random.randint(-20, 20, (canvas_size, canvas_size, 3))
        solid_color = np.full((canvas_size, canvas_size, 3), color, dtype=np.int16)
        
        # Rengi uygula ve uint8'e çevir
        textured_color = np.clip(solid_color + noise, 0, 255).astype(np.uint8)

        # --- GÖVDEYİ OLUŞTUR ---
        temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(temp_mask, [pts], 255)
        
        # Hafif şişirme (Eti dolgun olsun)
        # Profil çok küçükse (20 pikselden azsa) şişirme yapma
        if size < 20:
             dilate_kernel = np.ones((1, 1), np.uint8)
        else:
             dilate_kernel = np.ones((2, 2), np.uint8)
             
        thick_mask = cv2.dilate(temp_mask, dilate_kernel, iterations=1)
        
        # Gövde rengini maskeye uygula
        img = cv2.bitwise_and(textured_color, textured_color, mask=thick_mask)

        # --- DÜZELTME 3: KONTÜR (ÇİZGİ) GERİ GELDİ ---
        # Ama sadece 1 piksel kalınlığında ve çok siyah değil (Koyu Gri)
        # Siyah (0,0,0) yaparsak çok sert duruyor, (50,50,50) gibi yapıyoruz.
        edge_val = int(base_gray * 0.3) # Gövdenin %30'u kadar koyu
        edge_color = (edge_val, edge_val, edge_val)
        
        # Çizgiyi orijinal 'pts' üzerine çiziyoruz ki detay kaybolmasın.
        cv2.polylines(img, [pts], True, edge_color, 1, cv2.LINE_AA)

        # Kırpma
        coords = cv2.findNonZero(thick_mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return img[y:y+h, x:x+w], (x, y, w, h)
        return img, (0,0,0,0)

# --- 3. PALET ÜRETİCİ (GÜNCELLENDİ: AKILLI PROFİL MANTIĞI) ---
class SyntheticPalletGenerator:
    def __init__(self, profile_loader, renderer, bg_folder='background'):
        self.loader = profile_loader
        self.renderer = renderer
        self.pallet_size = (640, 640)
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']
        self.bg_images = []
        for ext in extensions:
            self.bg_images.extend(glob.glob(os.path.join(bg_folder, ext)))
            
        if not self.bg_images:
            print(f"UYARI: '{bg_folder}' boş. Siyah ekran kullanılacak.")

    def get_random_background(self):
        if self.bg_images:
            bg_path = random.choice(self.bg_images)
            bg = cv2.imread(bg_path)
            if bg is not None:
                bg = cv2.resize(bg, self.pallet_size)
                bg = (bg * random.uniform(0.3, 0.8)).astype(np.uint8)
                return bg
        
        bg = np.ones((640, 640, 3), dtype=np.uint8) * random.randint(20, 50)
        noise = np.random.randint(-10, 10, (640, 640, 3))
        return np.clip(bg + noise, 0, 255).astype(np.uint8)
    
    def generate_pallet(self):
        pallet = self.get_random_background()
        annotations = []
        
        if not self.loader.profiles: return pallet, [], {}

        # %15 Boş Örnek
        if random.random() < 0.15:
            return pallet, [], {}

        ptype = random.choice(list(self.loader.profiles.keys()))
        profile_contour = self.loader.profiles[ptype]
        
        base_size = random.randint(12, 35)
        scene_tint = random.choice(['neutral', 'neutral', 'yellow', 'blue'])
        
        # --- ÖZEL DAVRANIŞ ---
        is_nesting_type = False
        base_rotation_angle = 0
        
        if '7170' in ptype:
            is_nesting_type = True
            base_rotation_angle = 90
        else:
            is_nesting_type = False
            test_img, _ = self.renderer.render_profile(profile_contour, base_size, 0)
            h_test, w_test = test_img.shape[:2]
            if h_test > w_test:
                base_rotation_angle = 90
            else:
                base_rotation_angle = 0

        ref_img, _ = self.renderer.render_profile(profile_contour, base_size, base_rotation_angle)
        ph, pw = ref_img.shape[:2]
        if ph == 0 or pw == 0: return pallet, [], {}

        # --- GÜNCELLEME: RANDOM RAF KARANLIĞI ---
        overlay = pallet.copy()
        cv2.rectangle(overlay, (0, 0), (self.pallet_size[0], self.pallet_size[1]), (5, 5, 5), -1)
        
        # Kararma oranı: 0.60 ile 0.95 arası (Çoğunlukla çok karanlık)
        # 0.95 = Neredeyse zifiri siyah
        # 0.60 = Arkadaki fabrika hayal meyal görünüyor
        darkness_level = random.uniform(0.60, 0.95)
        
        cv2.addWeighted(overlay, darkness_level, pallet, 1 - darkness_level, 0, pallet)

        current_y = self.pallet_size[1] - 5 
        
        while current_y > 10:
            if random.random() < 0.6: 
                spacer = 0 
            else:
                spacer = random.randint(10, 25)
                
            row_y = current_y - spacer - ph
            if row_y < 0: break
            
            current_x = random.randint(0, 30)
            row_flip = random.choice([0, 180]) 
            
            while current_x < self.pallet_size[0] - pw:
                cur_size = int(base_size * random.uniform(0.95, 1.05))
                jitter = random.uniform(-2, 2)
                
                if is_nesting_type:
                    rotation = base_rotation_angle + row_flip + jitter
                else:
                    piece_flip = random.choice([0, 180])
                    rotation = base_rotation_angle + piece_flip + jitter
                
                img, _ = self.renderer.render_profile(profile_contour, cur_size, rotation, scene_tint)
                h, w = img.shape[:2]
                if h==0 or w==0: 
                    current_x += pw; continue
                
                pos_x = current_x
                pos_y = row_y + random.randint(-1, 1)
                
                if pos_x + w > self.pallet_size[0]: break
                
                self.blend_profile(pallet, img, pos_x, pos_y)
                
                cx = (pos_x + w/2) / self.pallet_size[0]
                cy = (pos_y + h/2) / self.pallet_size[1]
                
                class_id = list(self.loader.profiles.keys()).index(ptype)
                annotations.append({'class_id': class_id, 'bbox': [cx, cy, w/640, h/640]})
                
                if is_nesting_type:
                    overlap_ratio = random.uniform(0.15, 0.25)
                    overlap_px = int(w * overlap_ratio)
                    current_x += (w - overlap_px)
                else:
                    gap = 0 if random.random() < 0.8 else random.randint(1, 3)
                    current_x += (w + gap)
            
            current_y = row_y 

        return pallet, annotations, {}
        
    def blend_profile(self, pallet, profile, x, y):
            h, w = profile.shape[:2]
            if y < 0 or x < 0 or y+h > pallet.shape[0] or x+w > pallet.shape[1]: return
            
            roi = pallet[y:y+h, x:x+w]
            
            # --- YENİLİK: DİKEY GRADYAN (VERTICAL GRADIENT) ---
            # Kutucuk oluşmasın diye dört tarafı şişirmiyoruz.
            # Sadece profilin bulunduğu yere hafif bir "atmosferik ışık" veriyoruz.
            
            # Profil maskesi
            img2gray = cv2.cvtColor(profile, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
            
            # Maskeyi çok hafif yumuşat (kenarlar keskin kalmasın)
            soft_mask = cv2.GaussianBlur(mask, (3, 3), 0)
            
            # Işık efekti için maskeyi dikey olarak kaydıracağız
            # light_dir = -1 (Üst aydınlık), light_dir = 1 (Alt aydınlık)
            shift_amount = 2 
            
            # ROI üzerinde çalışacağız
            roi_float = roi.astype(float)
            
            # Gradyan mantığı: Profilin hemen dibinde hafif bir grilik olsun
            # ama bu grilik kutu gibi değil, sadece kenarlarda olsun.
            
            # Basit bir yöntem: Profilin olduğu yeri çok hafif aydınlat (Metalin yansıması)
            # Ama arka plan zaten siyah olduğu için bu yeterli olacaktır.
            
            # Profili yapıştıracağımız yerin arkasını %20 daha karart (ekstra derinlik)
            # (Zaten karanlık ama profilin tam arkası zifiri olsun)
            # shadow_mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=1)
            # shadow_factor = cv2.merge([shadow_mask, shadow_mask, shadow_mask]) / 255.0
            # roi_float = roi_float * (1.0 - (shadow_factor * 0.3))
            
            roi[:] = roi_float.astype(np.uint8)

            # --- PROFİLİ YAPIŞTIR ---
            mask_inv = cv2.bitwise_not(mask)
            
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv2.bitwise_and(profile, profile, mask=mask)
            
            dst = cv2.add(img1_bg, img2_fg)
            pallet[y:y+h, x:x+w] = dst

def apply_clahe(img):
    # Lab renk uzayına çevir (Parlaklığı renkten ayırmak için)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE uygula (Sadece L - Lightness kanalına)
    # clipLimit: Kontrast limiti (yüksekse çok sert olur, 2.0-4.0 iyidir)
    # tileGridSize: Izgara boyutu (küçük detaylar için 8x8 ideal)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Kanalları birleştir ve BGR'ye geri dön
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

# --- 4. ANA ÇALIŞTIRMA ---
def main():
    dxf_folder = '.' 
    bg_folder = 'background' 
    
    dxf_files = glob.glob(os.path.join(dxf_folder, "*.dxf"))
    loader = DXFProfileLoader()
    print(f"DXF dosyaları aranıyor: {dxf_files}")
    
    for f in dxf_files: loader.load_dxf(f, os.path.splitext(os.path.basename(f))[0])
    
    if not loader.profiles: 
        print("DXF bulunamadı!")
        return

    renderer = ProfileRenderer()
    generator = SyntheticPalletGenerator(loader, renderer, bg_folder)
    
    base_dir = "dataset_final"
    for d in ['images/train', 'labels/train', 'images/val', 'labels/val']:
        os.makedirs(f'{base_dir}/{d}', exist_ok=True)
        
    print(f"\nSentetik veri üretiliyor (V6 - Profil Bazlı Akıllı Dizilim)...")
    
    # 400 resim üretelim, negatif örnekler için sayı artsın
    for i in range(IMAGE_COUNT):
        if i % 50 == 0: print(f"İlerleme: {i}/400")
        pallet, anns, _ = generator.generate_pallet()

        clahe_pallet = apply_clahe(pallet)
        
        cv2.imwrite(f'{base_dir}/images/train/syn_{i}.jpg', pallet)
        cv2.imwrite(f'{base_dir}/images/train/syn_clahe{i}.jpg', clahe_pallet)
        with open(f'{base_dir}/labels/train/syn_{i}.txt', 'w') as f:
            for a in anns:
                f.write(f"{a['class_id']} {a['bbox'][0]:.6f} {a['bbox'][1]:.6f} {a['bbox'][2]:.6f} {a['bbox'][3]:.6f}\n")
    
    with open(f'{base_dir}/data.yaml', 'w') as f:
        names = "\n".join([f"  {i}: {n}" for i, n in enumerate(loader.profiles.keys())])
        f.write(f"path: {os.path.abspath(base_dir)}\ntrain: images/train\nval: images/train\nnames:\n{names}\nnc: {len(loader.profiles)}")

    print(f"\nBitti! Kontrol et.")

if __name__ == "__main__":
    main()