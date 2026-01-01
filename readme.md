

# 🏭 Synthetic Aluminum Profile Dataset Generator

Bu proje, YOLOv8 tabanlı nesne tespiti modelleri için DXF çizimlerinden otomatik sentetik veri seti üreten bir Python aracıdır. Kod, teknik çizimleri okuyarak rastgele varyasyonlarla istiflenmiş profil görüntüleri ve bunlara ait etiketleri (annotations) oluşturur.

## 📋 Proje Hakkında

Bu yazılım, manuel veri etiketleme maliyetini düşürmek amacıyla geliştirilmiştir. Sistem, `.dxf` formatındaki CAD verilerini okuyarak bunları görüntüye çevirir (rasterize eder) ve sentetik arka planlar üzerine yerleştirir. Ayrıca, modelin yanlış pozitif (false positive) üretimini engellemek amacıyla, içinde profil bulunmayan "negatif örnekleri" eğitim setine dahil etme yeteneğine sahiptir.

## 📂 Dosya Yapısı ve Modüller

Proje 4 ana bileşenden oluşmaktadır:

### 1. `DXFProfileLoader`

* **İşlevi:** Verilen klasördeki `.dxf` dosyalarını tarar.
* **Yaptıkları:** `ezdxf` kütüphanesini kullanarak LINE, POLYLINE, LWPOLYLINE, ARC ve CIRCLE objelerini okur. Bu vektörel verileri kapalı birer nokta kümesine (numpy array) dönüştürür.
* **Çıktı:** Normalize edilmiş (0-1 aralığında) profil kontürleri.

### 2. `ProfileRenderer`

* **İşlevi:** Vektörel profil verisini piksel tabanlı görüntüye (raster) çevirir.
* **Yaptıkları:**
* Belirtilen boyut ve açıda profili çizer.
* Basit renk ataması (gri tonlar) ve piksel gürültüsü (noise) ekleyerek metalik bir görünüm simülasyonu yapar.
* Sarı ve mavi tonlamalar (tint) ekleyerek varyasyon yaratır.



### 3. `SyntheticPalletGenerator`

* **İşlevi:** Profillerin bir palet üzerinde nasıl duracağını belirler.
* **Yaptıkları:**
* **Izgara (Grid) Mantığı:** Profilleri aşağıdan yukarıya doğru, satır satır dizer.
* **Rastgelelik:** Her profil için boyut (scale), dönme açısı (rotation) ve konum (jitter) değerlerini rastgele değiştirir.
* **Arka Plan:** `background/` klasöründen rastgele bir resim seçer veya gürültülü gri bir zemin oluşturur.
* **Etiketleme:** Yerleştirilen her profilin merkez koordinatlarını ve boyutlarını YOLO formatında (class, x, y, w, h) hesaplar.



### 4. `main` (Orkestrasyon)

* **İşlevi:** Tüm süreci yönetir ve dosyaları diske yazar.
* **Yaptıkları:**
* Belirlenen sayıda (varsayılan: 300) sentetik resim üretir.
* Her resme CLAHE (Kontrast Dengeleme) uygulayarak ikinci bir varyasyonunu kaydeder.
* `false_samples` klasöründeki görüntüleri okur ve bunları boş `.txt` dosyalarıyla veri setine ekler (Background Images).
* Eğitim için gerekli olan `data.yaml` dosyasını otomatik oluşturur.



---

## 🛠 Mevcut Durum ve Teknik Detaylar

Kodun şu anki versiyonunda uygulanan yöntemler ve mevcut sınırlamalar aşağıdadır:

### ✅ Uygulanan Özellikler

* **Veri Okuma:** 2D DXF dosyaları desteklenmektedir.
* **Görüntü İşleme:** OpenCV kullanılarak temel çizim, döndürme ve ölçekleme işlemleri yapılmaktadır.
* **Veri Artırma (Augmentation):** Renk tonu değişimi, boyut değişimi ve CLAHE filtresi uygulanmaktadır.
* **Entegrasyon:** Çıktılar direkt olarak YOLOv8 eğitimine uygun klasör yapısında (`images/train`, `labels/train`) üretilmektedir.

### ❌ Uygulanmamış Özellikler / Sınırlamalar

* **Fizik Simülasyonu:** Profillerin yerleşimi fizik kurallarına (yerçekimi, çarpışma) dayanmaz. Basit bir matematiksel döngü ile üst üste (overlap) bindirilerek yerleştirilir.
* **Doku (Texture):** Gerçek metal dokusu veya yüzey kusurları (çizik, pas) kullanılmamaktadır. Sadece rastgele piksel gürültüsü (random noise) mevcuttur.
* **Işıklandırma:** 3D ışıklandırma, gölge düşürme (drop shadow) veya yansıma efektleri yoktur. Görüntüler 2D ve düzdür.
* **Perspektif:** Kamera açısı simülasyonu yapılmamaktadır. Tüm çizimler tam karşıdan (ortografik) görünüm şeklindedir.

---

## 🚀 Kurulum

Gerekli Python kütüphaneleri:

```bash
pip install ezdxf opencv-python numpy

```

## ▶️ Kullanım

1. `.dxf` dosyalarınızı proje ana dizinine yerleştirin.
2. (Opsiyonel) `background/` klasörüne zemin görselleri ekleyin.
3. (Opsiyonel) `false_samples/` klasörüne, modelin öğrenmesi istenen boş (profilsiz) görselleri ekleyin.
4. Scripti çalıştırın:
```bash
python generator.py

```



Kod çalıştırıldığında `dataset_false_24_12` (veya kodda belirtilen isimde) bir klasör oluşturacak ve verileri oraya kaydedecektir.