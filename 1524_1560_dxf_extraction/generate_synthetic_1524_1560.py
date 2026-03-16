import ezdxf
import cv2
import numpy as np
import os
import networkx as nx
import random

# --- AYARLAR ---
DXF_SQUARE = "../cad_files/1524_CLEAN.dxf"
DXF_RECT   = "../cad_files/1560_CLEAN.dxf"
OUTPUT_DIR = "final_fit_proofs"
RENDER_SIZE = 60 

# --- İNCE AYARLAR ---
# Kare profilde boşluk fazla demiştin, onu sıkılaştırıyoruz.
# 1.0 = Tam kalınlık kadar kaydır.
# 0.7 = Kalınlığın %70'i kadar kaydır (Daha iç içe geçer).
TIGHTNESS_SQUARE = 0.70  # <-- KAREYİ SIKILAŞTIRDIK
TIGHTNESS_RECT   = 1.00  # <-- DİKDÖRTGEN AYNI KALDI

# İstifleme Boşluğu
GRID_GAP = 0 # <-- ARTIK BOŞLUK YOK

class FinalTuner:
    def load_ordered_contour(self, dxf_path):
        if not os.path.exists(dxf_path): return None
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        G = nx.Graph()
        def to_key(p): return (round(p[0], 3), round(p[1], 3))
        for e in msp:
            if e.dxftype() == 'LINE':
                G.add_edge(to_key(e.dxf.start), to_key(e.dxf.end))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points('xy')
                for i in range(len(pts)-1): G.add_edge(to_key(pts[i]), to_key(pts[i+1]))
                if e.closed: G.add_edge(to_key(pts[-1]), to_key(pts[0]))
        try:
            components = list(nx.connected_components(G))
            largest_comp = max(components, key=len)
            subgraph = G.subgraph(largest_comp)
            ordered_nodes = list(nx.dfs_preorder_nodes(subgraph))
            pts = np.array(ordered_nodes)
            min_vals, max_vals = np.min(pts, axis=0), np.max(pts, axis=0)
            center = (min_vals + max_vals) / 2
            pts = pts - center
            h = max_vals[1] - min_vals[1]
            return pts / h if h > 0 else pts
        except: return None

    def apply_aluminum_texture(self, mask):
        h, w = mask.shape
        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        base_val = random.randint(190, 230) # Parlak alüminyum
        
        # Işık Yansıması
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
        
        # İnce Kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, (80, 80, 80), 1, cv2.LINE_AA)
        return cv2.merge((img_bgr, mask))

    def get_single_profile(self, contour, size, rotation=0):
        canvas_size = int(size * 2.0)
        scaled = contour * size + canvas_size // 2
        if rotation != 0:
            M = cv2.getRotationMatrix2D((canvas_size//2, canvas_size//2), rotation, 1.0)
            ones = np.ones((len(scaled), 1))
            scaled = M.dot(np.hstack([scaled, ones]).T).T
        pts = scaled.astype(np.int32)
        
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # Köşe Yumuşatma (Radius)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        coords = cv2.findNonZero(mask)
        if coords is None: return None, 0
        x, y, w, h = cv2.boundingRect(coords)
        cropped_mask = mask[y:y+h, x:x+w]
        
        thickness = int(w * 0.12)
        textured_sprite = self.apply_aluminum_texture(cropped_mask)
        return textured_sprite, thickness

    def create_tight_block(self, contour, tightness_factor, config_type="X+Y+"):
        # Config: X+t Y+t varsayıyoruz (Senin beğendiğin)
        # Ama tightness_factor ile boşluğu ayarlıyoruz.
        
        sprite_a, t = self.get_single_profile(contour, RENDER_SIZE, 0)
        h, w = sprite_a.shape[:2]
        
        sprite_b_raw, _ = self.get_single_profile(contour, RENDER_SIZE, 180)
        sprite_b = cv2.resize(sprite_b_raw, (w, h))

        canvas_w = w * 2 + t * 4
        canvas_h = h * 2 + t * 4
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        cx, cy = canvas_w // 2, canvas_h // 2
        
        # A'yı Koy
        pos_a_x = cx - w // 2
        pos_a_y = cy - h // 2
        
        # Helper Paste
        def paste(bg, fg, x, y):
            fh, fw = fg.shape[:2]
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                bg[y:y+fh, x:x+fw, c] = (fg[:,:,c] * alpha + bg[y:y+fh, x:x+fw, c] * (1-alpha))
            bg[y:y+fh, x:x+fw, 3] = np.maximum(bg[y:y+fh, x:x+fw, 3], fg[:,:,3])
            return bg

        canvas = paste(canvas, sprite_a, pos_a_x, pos_a_y)
        
        # B'yi Kaydırarak Koy (SIKILIK AYARI BURADA)
        # X+t Y+t demiştik
        shift_val = int(t * tightness_factor)
        
        # "Kum saati" olmasın diye senin beğendiğin X+ Y+ yönünü kullanıyorum
        # Ama kare profil için belki X- Y+ daha iyidir? 
        # Senin beğendiğin "x:+t y:+t" idi, o yüzden + kullanıyorum.
        
        pos_b_x = pos_a_x + shift_val
        pos_b_y = pos_a_y + shift_val 
        
        # Eğer karede çok ayrık duruyorsa, X'i negatife çekmek gerekebilir.
        # Ama sen X+ Y+ beğendin, sadece "boşluk var" dedin.
        # O yüzden yönü değiştirmeden sadece mesafeyi kısalttım (shift_val).
        
        canvas = paste(canvas, sprite_b, pos_b_x, pos_b_y)
        
        alpha = canvas[:, :, 3]
        coords = cv2.findNonZero(alpha)
        x, y, cw, ch = cv2.boundingRect(coords)
        return canvas[y:y+ch, x:x+cw]

    def generate_wall(self, contour, name, rows, cols, tightness):
        print(f"🧱 Duvar Örülüyor: {name} ({cols}x{rows}) | Sıkılık: {tightness}")
        
        # Tek bir örnek blok üret (Hız için)
        block = self.create_tight_block(contour, tightness)
        bh, bw = block.shape[:2]
        
        # Duvar Tuvali (BOŞLUKSUZ)
        wall_w = cols * bw
        wall_h = rows * bh
        wall_img = np.ones((wall_h, wall_w, 3), dtype=np.uint8) * 240
        
        for r in range(rows):
            for c in range(cols):
                # Her seferinde yeniden üret ki dokular (texture) farklı olsun,
                # böylece yan yana gelince çizgi gibi durmasın, ayrı blok olduğu belli olsun.
                current_block = self.create_tight_block(contour, tightness)
                
                # Konum (Gap = 0)
                pos_x = c * bw
                pos_y = r * bh
                
                # Taşan kısımları kırp (Güvenlik)
                if pos_x + bw > wall_w or pos_y + bh > wall_h: continue
                
                # Overlay
                alpha = current_block[:, :, 3] / 255.0
                roi = wall_img[pos_y:pos_y+bh, pos_x:pos_x+bw]
                
                for ch in range(3):
                    roi[:, :, ch] = (current_block[:, :, ch] * alpha + roi[:, :, ch] * (1-alpha))
        
        filename = f"{OUTPUT_DIR}/{name}_WALL_{cols}x{rows}.png"
        cv2.imwrite(filename, wall_img)
        print(f"✅ Kaydedildi: {filename}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tuner = FinalTuner()
    
    # KARE (1524) - Sıkılaştırılmış
    cnt_sq = tuner.load_ordered_contour(DXF_SQUARE)
    if cnt_sq is not None:
        tuner.generate_wall(cnt_sq, "1524", 4, 10, TIGHTNESS_SQUARE)
        tuner.generate_wall(cnt_sq, "1524", 6, 8, TIGHTNESS_SQUARE)

    # DİKDÖRTGEN (1560) - Normal
    cnt_rect = tuner.load_ordered_contour(DXF_RECT)
    if cnt_rect is not None:
        tuner.generate_wall(cnt_rect, "1560", 4, 10, TIGHTNESS_RECT)
        tuner.generate_wall(cnt_rect, "1560", 6, 8, TIGHTNESS_RECT)

if __name__ == "__main__":
    main()