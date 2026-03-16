import ezdxf
import networkx as nx
import math
import os

# --- AYARLAR ---
INPUT_DXF = "../cad_files/1524.dxf"  # Temizlenecek dosya
OUTPUT_DXF = "clean_1524.dxf" # Kaydedilecek temiz dosya

def clean_dxf_geometry(input_path, output_path):
    if not os.path.exists(input_path):
        print("❌ Dosya bulunamadı.")
        return

    print(f"📂 Analiz ediliyor: {input_path}")
    try:
        doc = ezdxf.readfile(input_path)
        msp = doc.modelspace()
    except Exception as e:
        print(f"Hata: {e}")
        return

    # 1. Tüm Çizgileri Topla (Explode mantığı)
    # Her çizginin başlangıç ve bitiş noktalarını bir grafiğe düğüm olarak ekleyeceğiz.
    G = nx.Graph()
    
    line_count = 0
    
    # Koordinatları yuvarlayarak float hatalarını önleyelim (Tolerans)
    def to_key(point):
        return (round(point[0], 2), round(point[1], 2)) # 2 basamak hassasiyet

    entities_map = {} # Edge -> Entity listesi

    for entity in msp:
        points = []
        if entity.dxftype() == 'LINE':
            points = [entity.dxf.start, entity.dxf.end]
        elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            if entity.dxftype() == 'LWPOLYLINE':
                pts = list(entity.get_points('xy'))
            else:
                pts = [v.dxf.location for v in entity.vertices]
            
            # Polylines'ı parça parça kenarlara böl
            for i in range(len(pts) - 1):
                p1, p2 = to_key(pts[i]), to_key(pts[i+1])
                G.add_edge(p1, p2)
                line_count += 1
            
            if entity.is_closed: # Kapalıysa sonu başa bağla
                p1, p2 = to_key(pts[-1]), to_key(pts[0])
                G.add_edge(p1, p2)
                line_count += 1
            continue # Polyline işlendi, döngüye devam

        elif entity.dxftype() == 'CIRCLE':
            # Çemberleri direk aday olarak alalım (Merkez noktasıyla temsil edelim şimdilik)
            # Ama genelde profil kesitleri circle olmaz, line olur.
            pass
            
        if len(points) == 2:
            p1 = to_key(points[0])
            p2 = to_key(points[1])
            if p1 != p2: # Nokta değilse
                G.add_edge(p1, p2)
                line_count += 1

    print(f"🧩 Toplam {line_count} çizgi parçası bulundu.")
    
    if line_count == 0:
        print("❌ Çizim verisi bulunamadı!")
        return

    # 2. Bağlantılı Bileşenleri Bul (Islands)
    # Birbirine değen çizgiler bir grup oluşturur.
    components = list(nx.connected_components(G))
    print(f"🏝️ {len(components)} farklı parça (ada) bulundu.")

    # 3. Bileşenleri Analiz Et (Boyutlarına Göre)
    candidates = []
    
    all_nodes = list(G.nodes)
    # Tüm çizimin sınırları (Frame tespiti için)
    global_min_x = min(n[0] for n in all_nodes)
    global_max_x = max(n[0] for n in all_nodes)
    global_min_y = min(n[1] for n in all_nodes)
    global_max_y = max(n[1] for n in all_nodes)
    global_width = global_max_x - global_min_x
    global_height = global_max_y - global_min_y
    global_diag = math.sqrt(global_width**2 + global_height**2)

    print(f"📏 Tuval Boyutu: {global_width:.2f} x {global_height:.2f}")

    for idx, comp in enumerate(components):
        # Bu parçanın sınırlarını bul
        xs = [n[0] for n in comp]
        ys = [n[1] for n in comp]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        diag = math.sqrt(w**2 + h**2)
        
        # --- KRİTİK FİLTRELER ---
        
        # 1. Çok küçükleri at (Yazılar, noktalar)
        if diag < global_diag * 0.05: # Tuvalin %5'inden küçükse çöptür
            continue
            
        # 2. Çerçeve mi? (Tuval ile neredeyse aynı boyuttaysa çerçevedir)
        is_frame = False
        if w > global_width * 0.90 and h > global_height * 0.90:
            is_frame = True
            
        candidates.append({
            'id': idx,
            'nodes': list(comp),
            'diag': diag,
            'is_frame': is_frame,
            'center': (sum(xs)/len(xs), sum(ys)/len(ys))
        })

    # 4. En İyi Adayı Seç
    # Çerçeve olmayan EN BÜYÜK parçayı arıyoruz.
    valid_candidates = [c for c in candidates if not c['is_frame']]
    
    if not valid_candidates:
        print("⚠️ Uyarı: Sadece çerçeve bulundu veya her şey çok küçük.")
        # Çerçeve dışında bir şey yoksa, belki çerçeve sandığımız şey profildir?
        # En büyük 2. adayı almayı deneyelim (Eğer varsa)
        candidates.sort(key=lambda x: x['diag'], reverse=True)
        best_candidate = candidates[0] 
        print("   -> En büyük parça seçiliyor (Frame olabilir).")
    else:
        # Büyüklüğe göre sırala
        valid_candidates.sort(key=lambda x: x['diag'], reverse=True)
        best_candidate = valid_candidates[0]
        print(f"✅ Profil Bulundu! (ID: {best_candidate['id']}, Boyut: {best_candidate['diag']:.2f})")
        
        if len(valid_candidates) > 1:
             print(f"   (Diğer adaylar elendi: {len(valid_candidates)-1} adet)")

    # 5. Temiz DXF Oluştur
    new_doc = ezdxf.new()
    new_msp = new_doc.modelspace()
    
    # Seçilen adadaki noktalar arasındaki kenarları bul ve çiz
    # G grafiğinden bu düğümler arasındaki kenarları alacağız
    subgraph = G.subgraph(best_candidate['nodes'])
    
    for u, v in subgraph.edges():
        new_msp.add_line(u, v)
        
    new_doc.saveas(output_path)
    print(f"💾 Temiz dosya kaydedildi: {output_path}")
    print("👉 Şimdi test_interlock.py veya debug_dxf.py ile bu YENİ dosyayı test et.")

if __name__ == "__main__":
    # Gerekli kütüphaneyi kur: pip install networkx
    clean_dxf_geometry(INPUT_DXF, OUTPUT_DXF)