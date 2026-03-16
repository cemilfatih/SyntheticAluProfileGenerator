import ezdxf
import networkx as nx
import os

# --- HEDEFLER (Dosya Yolu -> Hedef ID) ---
TARGETS = {
    "../cad_files/1524.dxf": 3872,  # Karemsi olan
    "../cad_files/1560.dxf": 3849   # Dikdörtgenimsi olan
}

def extract_and_clean():
    for dxf_path, target_id in TARGETS.items():
        if not os.path.exists(dxf_path):
            print(f"❌ Dosya bulunamadı: {dxf_path}")
            continue

        print(f"Surgery Başlıyor: {dxf_path} (Hedef ID: {target_id})")
        
        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
        except Exception as e:
            print(f"   DXF Okuma Hatası: {e}")
            continue

        # 1. Grafiği Birebir Aynı Mantıkla Yeniden Oluştur
        # (Sıralamanın değişmemesi için inspect kodunun aynısını kullanıyoruz)
        G = nx.Graph()
        def to_key(p): return (round(p[0], 2), round(p[1], 2))

        print("   -> Grafik oluşturuluyor...")
        for entity in msp:
            points = []
            if entity.dxftype() == 'LINE':
                points = [entity.dxf.start, entity.dxf.end]
            elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                if entity.dxftype() == 'LWPOLYLINE': pts = list(entity.get_points('xy'))
                else: pts = [v.dxf.location for v in entity.vertices]
                for i in range(len(pts)-1): G.add_edge(to_key(pts[i]), to_key(pts[i+1]))
                if entity.is_closed: G.add_edge(to_key(pts[-1]), to_key(pts[0]))
                continue
            
            if len(points) == 2:
                G.add_edge(to_key(points[0]), to_key(points[1]))

        # 2. Parçaları Bul
        components = list(nx.connected_components(G))
        
        if target_id >= len(components):
            print(f"❌ HATA: ID {target_id} sınır dışı! (Toplam parça: {len(components)})")
            continue

        # 3. Hedef Parçayı Seç
        target_nodes = list(components[target_id])
        print(f"   -> Parça bulundu! ({len(target_nodes)} nokta içeriyor)")

        # 4. Merkeze Hizala (Centering)
        # Çizim muhtemelen uzay boşluğunda bir yerdedir, (0,0)'a taşıyalım.
        xs = [n[0] for n in target_nodes]
        ys = [n[1] for n in target_nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 5. Yeni Temiz DXF Oluştur
        new_doc = ezdxf.new()
        new_msp = new_doc.modelspace()
        
        # Sadece hedef parçanın kenarlarını (edge) çiz
        subgraph = G.subgraph(target_nodes)
        line_count = 0
        for u, v in subgraph.edges():
            # Merkeze taşıyarak ekle
            p1 = (u[0] - center_x, u[1] - center_y)
            p2 = (v[0] - center_x, v[1] - center_y)
            new_msp.add_line(p1, p2)
            line_count += 1
            
        # Kaydet
        dir_name = os.path.dirname(dxf_path)
        base_name = os.path.splitext(os.path.basename(dxf_path))[0]
        output_path = os.path.join(dir_name, f"{base_name}_CLEAN.dxf")
        
        new_doc.saveas(output_path)
        print(f"✅ TEMİZ DOSYA KAYDEDİLDİ: {output_path} ({line_count} çizgi)")

if __name__ == "__main__":
    extract_and_clean()