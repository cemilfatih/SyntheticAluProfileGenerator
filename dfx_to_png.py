import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import cv2
import numpy as np

def dxf_to_png_auto(dxf_path, output_path):
    print(f"DXF Dosyası Okunuyor: {dxf_path}...")
    
    try:
        # 1. DXF Yükle
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        # 2. Matplotlib Kurulumu
        # Arkaplanı siyah yapıyoruz (Model eğitimi için daha net kontrast)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0, 0, 1, 1])
        
        # Çizim Bağlamı
        ctx = RenderContext(doc)
        
        # Çiziciyi Ayarla
        out = MatplotlibBackend(ax)
        
        # 3. Çizimi Yap
        Frontend(ctx, out).draw_layout(msp, finalize=True)
        
        # 4. OTOMATİK ÖLÇEKLENDİRME (Manuel hesaplama yerine)
        ax.autoscale(enable=True, axis='both', tight=True)
        
        # Eksenleri kapat
        ax.axis('off')
        
        # 5. Geçici Kaydet
        fig.savefig('temp_auto.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # 6. Görüntüyü İşle (Siyah Arkaplan, Beyaz Profil)
        img = cv2.imread('temp_auto.png')
        if img is None:
            print("Görüntü oluşturulamadı.")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold (Otomatik algılasın)
        # Genelde Matplotlib çıktısında çizgiler siyahtır, arkaplan beyazdır.
        # Biz bunu terse çevireceğiz: Arkaplan SİYAH(0), Profil BEYAZ(255)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # İyileştirme (Çizgileri birleştir)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(thresh, kernel, iterations=2)
        
        # Fazla boşlukları kırp (Crop)
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Biraz marj bırak
            crop = mask[y:y+h, x:x+w]
            
            cv2.imwrite(output_path, crop)
            print(f"BAŞARILI! Profil maskesi oluşturuldu: {output_path}")
            print("Şimdi dataset_generator.py kodunu çalıştırabilirsin.")
        else:
            print("HATA: Çizim kaydedildi ama boş görünüyor.")

    except Exception as e:
        print(f"HATA: {e}")

# ÇALIŞTIR
dxf_to_png_auto('9794.dxf', 'profil_maskesi.png')