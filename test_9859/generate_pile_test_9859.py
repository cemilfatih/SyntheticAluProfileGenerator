import ezdxf
import cv2
import numpy as np
import os
import random
import math

# --- AYARLAR ---
DXF_PROFILE = "../cad_files/9859.dxf"
OUTPUT_DIR = "pallet_test_out"
RENDER_SIZE = 300  # Ram'i yormamak için biraz küçültüldü, 20 katmanlı devasa bir görsel çıkacak

class FinalTuner:
    def get_polygons_from_lines(self, lines, tolerance=3.0):
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
                        
                if best_idx != -1 and best_dist <= tolerance:
                    edge = edges.pop(best_idx)
                    if best_reverse:
                        current_poly.append(edge[0])
                    else:
                        current_poly.append(edge[1])
                    found = True
                else:
                    break
                    
            if len(current_poly) > 2:
                polygons.append(np.array(current_poly, dtype=np.int32))
            
        return polygons

    def extract_mask_from_dxf(self, dxf_path, canvas_size):
        if not os.path.exists(dxf_path): 
            return None
            
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        lines = []
        for e in msp:
            if e.dxftype() == 'LINE':
                lines.append(( (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y) ))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points('xy')
                for i in range(len(pts)-1): lines.append((pts[i], pts[i+1]))
                if e.closed: lines.append((pts[-1], pts[0]))
            elif e.dxftype() == 'ARC':
                c, r = e.dxf.center, e.dxf.radius
                sa, ea = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
                if ea < sa: ea += 2 * math.pi
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(sa, ea, 30)]
                for i in range(len(pts)-1): lines.append((pts[i], pts[i+1]))
            elif e.dxftype() == 'CIRCLE':
                c, r = e.dxf.center, e.dxf.radius
                pts = [[c.x + r * math.cos(a), c.y + r * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 60)]
                for i in range(len(pts)-1): lines.append((pts[i], pts[i+1]))
                    
        if not lines: return None
            
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
            
        polygons = self.get_polygons_from_lines(scaled_lines, tolerance=3.0)
        polygons = sorted(polygons, key=cv2.contourArea, reverse=True)
        
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        for poly in polygons:
            temp_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [poly], 255)
            mask = cv2.bitwise_xor(mask, temp_mask)
            
        return mask

    def apply_aluminum_texture(self, mask):
        h, w = mask.shape
        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        base_val = random.randint(180, 220) 
        
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
        cv2.drawContours(img_bgr, contours, -1, (60, 60, 60), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    def get_single_profile(self, dxf_path, size):
        canvas_size = int(size * 3.0)
        mask = self.extract_mask_from_dxf(dxf_path, canvas_size)
        if mask is None: return None
            
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(mask)
        if coords is None: return None
        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y+h, x:x+w]
        
        textured_sprite = self.apply_aluminum_texture(cropped_mask)
        
        # Kesinlikle DİKEY durmasını sağla (Boy > En)
        ch, cw = textured_sprite.shape[:2]
        if cw > ch:
            textured_sprite = cv2.rotate(textured_sprite, cv2.ROTATE_90_CLOCKWISE)
            
        return textured_sprite

    def create_package(self, sprite, profiles_per_pkg=6):
        """ Profilleri yatayda yan yana dizip bir 'paket' oluşturur """
        h, w = sprite.shape[:2]
        package_w = w * profiles_per_pkg
        package_h = h
        
        package_sprite = np.zeros((package_h, package_w, 4), dtype=np.uint8)
        
        for i in range(profiles_per_pkg):
            x_offset = i * w
            package_sprite[:, x_offset:x_offset+w] = sprite
            
        return package_sprite

    def overlay_image(self, bg, fg, x, y):
        fh, fw = fg.shape[:2]
        bh, bw = bg.shape[:2]
        
        x1, x2 = max(0, x), min(bw, x + fw)
        y1, y2 = max(0, y), min(bh, y + fh)
        
        fg_x1 = x1 - x
        fg_x2 = x2 - x
        fg_y1 = y1 - y
        fg_y2 = y2 - y
        
        if fg_x1 >= fg_x2 or fg_y1 >= fg_y2: return
        
        fg_crop = fg[fg_y1:fg_y2, fg_x1:fg_x2]
        bg_crop = bg[y1:y2, x1:x2]
        
        alpha = fg_crop[:, :, 3] / 255.0
        for c in range(3):
            bg_crop[:, :, c] = (alpha * fg_crop[:, :, c] + (1 - alpha) * bg_crop[:, :, c])

    def generate_complex_pile(self, dxf_path, num_rows=18, pkgs_per_row=6, pattern="SANDWICH"):
        """
        Gelişmiş İstif (Pile) Oluşturucu
        pattern = 'UNIFORM', 'ALTERNATING', veya 'SANDWICH'
        """
        print(f"\n🧱 İstif Üretiliyor... Pattern: {pattern} | {num_rows} Sıra x {pkgs_per_row} Paket")
        
        base_sprite = self.get_single_profile(dxf_path, RENDER_SIZE)
        if base_sprite is None: return
        
        pkg_img = self.create_package(base_sprite, profiles_per_pkg=6)
        pkg_h, pkg_w = pkg_img.shape[:2]
        
        # İstifin nihai genişliği (Örn: 6 paket yan yana)
        pile_width = pkgs_per_row * pkg_w
        
        # Büyük bir tuval açalım (Aşağıdan yukarıya dizeceğiz)
        canvas_h = (num_rows + 5) * max(pkg_h, pkg_w) 
        canvas_w = pile_width + 100
        wall_img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240
        
        # Dizmeye en alttan başla
        current_y = canvas_h - 50 
        start_x = 50 

        for row_idx in range(num_rows):
            # Hangi Pattern'e göre dizeceğiz?
            row_type = "HORIZONTAL"
            flip = False
            
            if pattern == "ALTERNATING":
                if row_idx % 2 == 1: flip = True
                
            elif pattern == "SANDWICH":
                # Ortadaki 1/3'lük dilimi DİKEY paketlerle doldur
                one_third = num_rows // 3
                if one_third <= row_idx < 2 * one_third:
                    row_type = "VERTICAL"
            
            # --- YATAY KATMAN ---
            if row_type == "HORIZONTAL":
                layer_h = pkg_h
                current_y -= layer_h # Bir katman yukarı çık
                
                for col in range(pkgs_per_row):
                    x = start_x + (col * pkg_w)
                    stamp = pkg_img
                    if flip:
                        stamp = cv2.rotate(stamp, cv2.ROTATE_180)
                    self.overlay_image(wall_img, stamp, x, current_y)
            
            # --- DİKEY KATMAN (Sandviç İçi) ---
            else:
                stamp = cv2.rotate(pkg_img, cv2.ROTATE_90_CLOCKWISE)
                layer_h = stamp.shape[0] # Paketin genişliği artık katman yüksekliği oldu
                current_y -= layer_h
                
                # Bu yatay genişliğe kaç tane dikey paket sığar?
                v_pkgs_per_row = pile_width // stamp.shape[1]
                
                # Tam ortalamak için offset hesapla
                offset_x = start_x + (pile_width - (v_pkgs_per_row * stamp.shape[1])) // 2
                
                for col in range(v_pkgs_per_row):
                    x = offset_x + (col * stamp.shape[1])
                    self.overlay_image(wall_img, stamp, x, current_y)

        # İşlem bitince devasa tuvaldeki sadece dolu olan yeri kırp (Auto-crop)
        final_img = wall_img[current_y - 50: canvas_h, :]
        
        filename = f"{OUTPUT_DIR}/PILE_{pattern}_{num_rows}rows.png"
        cv2.imwrite(filename, final_img)
        print(f"✅ Başarıyla Kaydedildi: {filename}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tuner = FinalTuner()
    
    # 1. Standart düz dizilim (15 sıra, 5 paket)
    tuner.generate_complex_pile(DXF_PROFILE, num_rows=15, pkgs_per_row=5, pattern="UNIFORM")
    
    # 2. Ters-Düz birleşik dizilim (18 sıra, 6 paket)
    tuner.generate_complex_pile(DXF_PROFILE, num_rows=18, pkgs_per_row=6, pattern="ALTERNATING")
    
    # 3. Arası Dikey (Cross-Stack) Sandviç dizilim (20 sıra, 6 paket)
    tuner.generate_complex_pile(DXF_PROFILE, num_rows=20, pkgs_per_row=6, pattern="SANDWICH")

if __name__ == "__main__":
    main()