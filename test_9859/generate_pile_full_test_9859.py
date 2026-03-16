import ezdxf
import cv2
import numpy as np
import os
import random
import math
import glob

# --- TEST AYARLARI ---
DXF_PROFILE = "../cad_files/9859.dxf"
BG_FOLDER = "background"
OUTPUT_DIR = "pallet_test_out"
RENDER_SIZE = 150  # Test için boyut

class TestTuner:
    def __init__(self, bg_folder):
        self.bg_images = glob.glob(os.path.join(bg_folder, "*.*"))

    def get_random_background(self, w, h):
        if self.bg_images and random.random() < 0.9: 
            bg_path = random.choice(self.bg_images)
            bg = cv2.imread(bg_path)
            if bg is not None:
                bg = cv2.resize(bg, (w, h))
                factor = random.uniform(0.5, 0.95) # Işık/karanlık oranı her seferinde değişir
                return (bg.astype(float) * factor).astype(np.uint8)
                
        # Resim yoksa rastgele dokulu bir arka plan oluştur
        val = random.randint(50, 110)
        bg = np.ones((h, w, 3), dtype=np.uint8) * val
        noise = np.random.randint(-25, 25, bg.shape)
        return np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)

    def get_polygons_from_lines(self, lines, tolerance=3.0):
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
                    if d1 < best_dist: best_dist, best_idx, best_reverse = d1, i, False
                    if d2 < best_dist: best_dist, best_idx, best_reverse = d2, i, True
                if best_idx != -1 and best_dist <= tolerance:
                    edge = edges.pop(best_idx)
                    current_poly.append(edge[0] if best_reverse else edge[1])
                else:
                    break
            if len(current_poly) > 2:
                polygons.append(np.array(current_poly, dtype=np.int32))
        return polygons

    def extract_mask_from_dxf(self, dxf_path, canvas_size):
        if not os.path.exists(dxf_path): return None
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
            
        all_pts = np.array([pt for p1, p2 in lines for pt in (p1, p2)])
        min_vals, max_vals = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
        max_dim = max(max_vals[0] - min_vals[0], max_vals[1] - min_vals[1])
        if max_dim == 0: max_dim = 1
        
        padding = int(canvas_size * 0.1)
        scale = (canvas_size - 2 * padding) / max_dim
        
        scaled_lines = []
        for p1, p2 in lines:
            x1, y1 = (p1[0] - min_vals[0]) * scale + padding, canvas_size - ((p1[1] - min_vals[1]) * scale + padding)
            x2, y2 = (p2[0] - min_vals[0]) * scale + padding, canvas_size - ((p2[1] - min_vals[1]) * scale + padding)
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
        
        # Alüminyumun parlaklığı ve kontrastı her profilde rastgele değişir
        is_shiny = random.random() < 0.3
        base_val = random.randint(200, 240) if is_shiny else random.randint(150, 190)
        contrast = 30 if is_shiny else 15
        
        if random.random() < 0.5:
            for y in range(h):
                # np.clip ile 255'in üzerine çıkmasını engelliyoruz
                v = int(np.clip(base_val - contrast * np.sin(y / h * 6.28), 0, 255)) 
                img_bgr[y, :, :] = (v, v, v)
        else:
            for x in range(w):
                # np.clip ile 255'in üzerine çıkmasını engelliyoruz
                v = int(np.clip(base_val - contrast * np.sin(x / w * 6.28), 0, 255))
                img_bgr[:, x, :] = (v, v, v)
        
        noise = np.random.randint(-12, 12, img_bgr.shape)
        img_bgr = np.clip(img_bgr.astype(int) + noise, 0, 255).astype(np.uint8)
        img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, (50, 50, 50), 1, cv2.LINE_AA)
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
        
        ch, cw = textured_sprite.shape[:2]
        if cw > ch:
            textured_sprite = cv2.rotate(textured_sprite, cv2.ROTATE_90_CLOCKWISE)
            
        return textured_sprite

    def create_package(self, sprite, profiles_per_pkg=6):
        h, w = sprite.shape[:2]
        package_w = w * profiles_per_pkg
        package_h = h
        package_sprite = np.zeros((package_h, package_w, 4), dtype=np.uint8)
        for i in range(profiles_per_pkg):
            x_offset = i * w
            package_sprite[:, x_offset:x_offset+w] = sprite
        return package_sprite

    def overlay_image_alpha(self, img, img_overlay, pos_x, pos_y, alpha_mask):
        canvas_h, canvas_w = img.shape[:2]
        ov_h, ov_w = img_overlay.shape[:2]
        pos_x, pos_y = int(pos_x), int(pos_y)
        
        x1, y1 = max(0, pos_x), max(0, pos_y)
        x2, y2 = min(canvas_w, pos_x + ov_w), min(canvas_h, pos_y + ov_h)
        if x2 <= x1 or y2 <= y1: return img
        
        ov_x1, ov_y1 = x1 - pos_x, y1 - pos_y
        ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)
        
        overlay_rgb = img_overlay[ov_y1:ov_y2, ov_x1:ov_x2, :3]
        mask_3ch = np.dstack([alpha_mask[ov_y1:ov_y2, ov_x1:ov_x2] / 255.0] * 3)
        bg_crop = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = (overlay_rgb * mask_3ch + bg_crop * (1.0 - mask_3ch)).astype(np.uint8)
        return img

    def generate_tunnel_void(self, w, h, obj_cx, obj_cy, canvas_w, canvas_h):
        screen_cx, screen_cy = canvas_w // 2, canvas_h // 2
        void_w, void_h = w - 4, h - 4
        if void_w <= 0 or void_h <= 0: return None
        
        vec_x, vec_y = screen_cx - obj_cx, screen_cy - obj_cy
        light_center_x = np.clip((void_w // 2) + int(vec_x * 0.3), 0, void_w)
        light_center_y = np.clip((void_h // 2) + int(vec_y * 0.3), 0, void_h)
        
        Y, X = np.ogrid[:void_h, :void_w]
        dist_map = np.sqrt((X - light_center_x)**2 + (Y - light_center_y)**2)
        max_dist = np.sqrt(void_w**2 + void_h**2)
        norm_dist = np.clip(dist_map / (max_dist * 0.6), 0, 1)
        
        # Gölgenin karanlık tonu da rastgele değişir
        alpha_channel = (np.power(norm_dist, 0.7) * random.uniform(0.80, 0.98) * 255).astype(np.uint8)
        base_color = np.zeros((void_h, void_w, 3), dtype=np.uint8)
        return cv2.merge([base_color[:,:,0], base_color[:,:,1], base_color[:,:,2], alpha_channel])

    def generate_test_pile(self, dxf_path, img_id, num_rows, pkgs_per_row, pattern):
        print(f"🧱 Çeşitlilik Testi #{img_id} | Pattern: {pattern} | {num_rows} Sıra, {pkgs_per_row} Paket/Sıra")
        
        base_sprite = self.get_single_profile(dxf_path, RENDER_SIZE)
        if base_sprite is None: return
        
        pkg_img = self.create_package(base_sprite, profiles_per_pkg=6)
        pkg_h, pkg_w = pkg_img.shape[:2]
        
        pile_width = pkgs_per_row * pkg_w
        canvas_w = pile_width + random.randint(300, 600) # Tuval genişliği de hafif esnek
        canvas_h = (num_rows + 2) * max(pkg_h, pkg_w) + 200
        
        wall_img = self.get_random_background(canvas_w, canvas_h)
        
        current_y = canvas_h - random.randint(80, 150) # Başlangıç yüksekliği esnek
        start_x = (canvas_w - pile_width) // 2 # İstifi her zaman ortala
        
        render_commands_depth = []
        render_commands_face = []

        for row_idx in range(num_rows):
            row_type = "HORIZONTAL"
            flip = False
            
            if pattern == "ALTERNATING" and row_idx % 2 == 1: flip = True
            elif pattern == "SANDWICH" and (num_rows // 3 <= row_idx < 2 * (num_rows // 3)): row_type = "VERTICAL"
            
            if row_type == "HORIZONTAL":
                layer_h = pkg_h
                current_y -= layer_h 
                for col in range(pkgs_per_row):
                    x = start_x + (col * pkg_w)
                    stamp = cv2.rotate(pkg_img, cv2.ROTATE_180) if flip else pkg_img
                    self._queue_render(render_commands_depth, render_commands_face, stamp, x, current_y, canvas_w, canvas_h)
            else:
                stamp = cv2.rotate(pkg_img, cv2.ROTATE_90_CLOCKWISE)
                layer_h = stamp.shape[0] 
                current_y -= layer_h
                v_pkgs_per_row = pile_width // stamp.shape[1]
                offset_x = start_x + (pile_width - (v_pkgs_per_row * stamp.shape[1])) // 2
                
                for col in range(v_pkgs_per_row):
                    x = offset_x + (col * stamp.shape[1])
                    self._queue_render(render_commands_depth, render_commands_face, stamp, x, current_y, canvas_w, canvas_h)

        for cmd in render_commands_depth:
            if cmd['dist'] > 0:
                nx, ny = cmd['vec_x']/cmd['dist'], cmd['vec_y']/cmd['dist']
                tox, toy = int(nx*cmd['depth_len']), int(ny*cmd['depth_len'])
                body_color = np.zeros((cmd['sh'], cmd['sw'], 3), dtype=np.uint8) + random.randint(30, 60)
                steps = min(int(max(abs(tox), abs(toy))), 20)
                for s in range(steps, 0, -1):
                    r = s/steps if steps>0 else 0
                    self.overlay_image_alpha(wall_img, body_color, cmd['pos_x']+int(tox*r), cmd['pos_y']+int(toy*r), cmd['alpha'])
            
            if cmd['void_sprite'] is not None:
                self.overlay_image_alpha(wall_img, cmd['void_sprite'], cmd['pos_x']+2, cmd['pos_y']+2, cmd['void_sprite'][:,:,3])

        for cmd in render_commands_face:
            self.overlay_image_alpha(wall_img, cmd['sprite'], cmd['pos_x'], cmd['pos_y'], cmd['sprite'][:,:,3])

        final_img = wall_img[max(0, current_y - 150): canvas_h, :]
        
        filename = f"{OUTPUT_DIR}/VAR_{img_id:02d}_{pattern}_{num_rows}Rx{pkgs_per_row}C.png"
        cv2.imwrite(filename, final_img)
        print(f"✅ Kaydedildi: {filename}")

    def _queue_render(self, depth_list, face_list, sprite, x, y, cw, ch):
        sh, sw = sprite.shape[:2]
        obj_cx, obj_cy = x + sw // 2, y + sh // 2
        vec_x, vec_y = (cw // 2) - obj_cx, (ch // 2) - obj_cy
        dist = math.sqrt(vec_x**2 + vec_y**2)
        max_dist = math.sqrt((cw/2)**2 + (ch/2)**2)
        depth_len = min((dist/max_dist)*60, sw*0.8)
        
        void_sprite = self.generate_tunnel_void(sw, sh, obj_cx, obj_cy, cw, ch)
        
        depth_list.append({
            'pos_x': x, 'pos_y': y, 'sw': sw, 'sh': sh, 'alpha': sprite[:,:,3],
            'vec_x': vec_x, 'vec_y': vec_y, 'dist': dist, 'depth_len': depth_len, 'void_sprite': void_sprite
        })
        face_list.append({
            'sprite': sprite, 'pos_x': x, 'pos_y': y
        })

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tuner = TestTuner(bg_folder=BG_FOLDER)
    
    print("\n--- ÇEŞİTLİLİK TESTİ BAŞLIYOR (10 FARKLI VARYASYON) ---")
    
    patterns = ["ALTERNATING", "SANDWICH", "UNIFORM"]
    # Ağırlıklar: %70 Alternating, %20 Sandwich, %10 Uniform
    weights = [0.70, 0.20, 0.10]
    
    for i in range(1, 11):
        # İstif özelliklerini rastgele seç
        chosen_pattern = random.choices(patterns, weights=weights, k=1)[0]
        random_rows = random.randint(8, 18)       # 8 ile 18 sıra arası
        random_cols = random.randint(3, 6)        # Yan yana 3 ile 6 paket arası
        
        tuner.generate_test_pile(
            dxf_path=DXF_PROFILE, 
            img_id=i, 
            num_rows=random_rows, 
            pkgs_per_row=random_cols, 
            pattern=chosen_pattern
        )

if __name__ == "__main__":
    main()