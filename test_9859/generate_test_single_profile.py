import ezdxf
import cv2
import numpy as np
import os
import random
import math

# --- AYARLAR ---
DXF_PROFILE = "../cad_files/9859.dxf"
OUTPUT_DIR = "single_dxf_test_out"
RENDER_SIZE = 400

class FinalTuner:
    def get_polygons_from_lines(self, lines, tolerance=3.0):
        """
        Dağınık line segmentlerini uç uca ekleyerek sıralı poligonlara (contour) dönüştürür.
        Boşlukların (gap) üzerinden atlamaz, tolerans mesafesi içindeki gerçek uçları takip eder.
        """
        edges = []
        for p1, p2 in lines:
            edges.append([np.array(p1), np.array(p2)])
            
        polygons = []
        
        while edges:
            current_poly = [edges[0][0], edges[0][1]]
            edges.pop(0)
            
            while True:
                last_pt = current_poly[-1]
                found = False
                
                best_dist = float('inf')
                best_idx = -1
                best_reverse = False
                
                # Kalan çizgiler içinde şu anki uca en yakın olanı bul
                for i, edge in enumerate(edges):
                    d1 = np.linalg.norm(edge[0] - last_pt)
                    d2 = np.linalg.norm(edge[1] - last_pt)
                    
                    if d1 < best_dist:
                        best_dist = d1
                        best_idx = i
                        best_reverse = False
                    if d2 < best_dist:
                        best_dist = d2
                        best_idx = i
                        best_reverse = True
                        
                # Eğer bulunan en yakın uç, toleransın altındaysa (gerçekten bitişikse) ekle
                if best_idx != -1 and best_dist <= tolerance:
                    edge = edges.pop(best_idx)
                    if best_reverse:
                        current_poly.append(edge[0])
                    else:
                        current_poly.append(edge[1])
                    found = True
                else:
                    break # Çizgi döngüsü kapandı veya gerçekten bir kopukluk var
                    
            if len(current_poly) > 2:
                polygons.append(np.array(current_poly, dtype=np.int32))
            
        return polygons

    def extract_mask_from_dxf(self, dxf_path, canvas_size):
        if not os.path.exists(dxf_path): 
            print(f"HATA: Dosya bulunamadı: {dxf_path}")
            return None
            
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        lines = []
        def add_line(p1, p2):
            lines.append(( (p1.x, p1.y), (p2.x, p2.y) ))

        for e in msp:
            if e.dxftype() == 'LINE':
                add_line(e.dxf.start, e.dxf.end)
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
            
        all_pts = []
        for p1, p2 in lines:
            all_pts.extend([p1, p2])
        all_pts = np.array(all_pts)
        min_vals = np.min(all_pts, axis=0)
        max_vals = np.max(all_pts, axis=0)
        
        w = max_vals[0] - min_vals[0]
        h = max_vals[1] - min_vals[1]
        max_dim = max(w, h)
        if max_dim == 0: max_dim = 1
        
        padding = int(canvas_size * 0.1)
        scale = (canvas_size - 2 * padding) / max_dim
        
        scaled_lines = []
        for p1, p2 in lines:
            x1 = (p1[0] - min_vals[0]) * scale + padding
            y1 = (p1[1] - min_vals[1]) * scale + padding
            x2 = (p2[0] - min_vals[0]) * scale + padding
            y2 = (p2[1] - min_vals[1]) * scale + padding
            y1 = canvas_size - y1
            y2 = canvas_size - y2
            scaled_lines.append( ((x1, y1), (x2, y2)) )
            
        # Parçaları birleştirip polygonlar haline getir
        polygons = self.get_polygons_from_lines(scaled_lines, tolerance=3.0)
        
        # Polygonları kapladıkları alana göre sırala (En büyük en önce - Outer Boundary)
        polygons = sorted(polygons, key=cv2.contourArea, reverse=True)
        
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        # XOR Mantığı (Dış sınır beyaza dolar, ardından içindeki delik gelince orayı sıfırlar)
        for poly in polygons:
            temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [poly], 255)
            # XOR işlemi: 0^255 = 255 (Doldur), 255^255 = 0 (Oy/Delik aç)
            mask = cv2.bitwise_xor(mask, temp_mask)
            
        return mask

    def apply_aluminum_texture(self, mask):
        h, w = mask.shape
        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        base_val = random.randint(190, 230) 
        
        if random.random() < 0.5:
            for y in range(h):
                v = int(base_val - 20 * np.sin(y / h * 6.28)) 
                img_bgr[y, :, :] = (v, v, v)
        else:
            for x in range(w):
                v = int(base_val - 20 * np.sin(x / w * 6.28))
                img_bgr[:, x, :] = (v, v, v)
        
        noise = np.random.randint(-10, 10, img_bgr.shape)
        img_bgr = np.clip(img_bgr.astype(int) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, (80, 80, 80), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    def get_single_profile(self, dxf_path, size, rotation=0):
        canvas_size = int(size * 3.0)
        mask = self.extract_mask_from_dxf(dxf_path, canvas_size)
        
        if mask is None:
            return None, 0
            
        if rotation != 0:
            M = cv2.getRotationMatrix2D((canvas_size//2, canvas_size//2), rotation, 1.0)
            mask = cv2.warpAffine(mask, M, (canvas_size, canvas_size))
            
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(mask)
        if coords is None:
            return None, 0
            
        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y+h, x:x+w]
        
        thickness = int(w * 0.12)
        textured_sprite = self.apply_aluminum_texture(cropped_mask)
        return textured_sprite, thickness

    def create_tight_block(self, dxf_path, tightness_factor, config_type="X+Y+"):
        sprite_a, t = self.get_single_profile(dxf_path, RENDER_SIZE, 0)
        if sprite_a is None:
            raise Exception("Profil oluşturulamadı!")
        return sprite_a

    def generate_wall(self, dxf_path, name, rows, cols, tightness):
        block = self.create_tight_block(dxf_path, tightness)
        bh, bw = block.shape[:2]
        
        wall_w = cols * bw
        wall_h = rows * bh
        wall_img = np.ones((wall_h, wall_w, 3), dtype=np.uint8) * 240
        
        for r in range(rows):
            for c in range(cols):
                current_block = self.create_tight_block(dxf_path, tightness)
                
                pos_x = c * bw
                pos_y = r * bh
                
                if pos_x + bw > wall_w or pos_y + bh > wall_h: continue
                
                alpha = current_block[:, :, 3] / 255.0
                roi = wall_img[pos_y:pos_y+bh, pos_x:pos_x+bw]
                
                for ch in range(3):
                    roi[:, :, ch] = (current_block[:, :, ch] * alpha + roi[:, :, ch] * (1-alpha))
        
        filename = f"{OUTPUT_DIR}/{name}_WALL_{cols}x{rows}.png"
        cv2.imwrite(filename, wall_img)
        print(f"✅ Başarıyla Kaydedildi: {filename}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tuner = FinalTuner()
    tuner.generate_wall(DXF_PROFILE, "9859", 4, 10, 1)

if __name__ == "__main__":
    main()