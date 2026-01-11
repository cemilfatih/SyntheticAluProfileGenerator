from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import cv2
import os
import numpy as np

# --- AYARLAR ---
MODEL_PATH = "models/11_01_yolo11s_fast_finetune.pt" 
CONF_THRESHOLD = 0.10          # Şimdilik 0.50 kalsın, skorları görüp artırırsın
SLICE_SIZE = 1280               
OVERLAP_RATIO = 0.20           
LINE_THICKNESS = 2             
DRAW_TEXT = True               # EVET, artık skorları yazıyor
MIN_PIXEL_SIZE = 10          # 10 pikselden küçük "leke" gibi kutuları çizme

test_folder_path = "test_images/"
out_folder_path = "out/output_images_10_01_yolo11s_fast_finetune/" # Klasör adını değiştirdim

def draw_boxes(image, prediction_result, output_path):
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    object_prediction_list = prediction_result.object_prediction_list
    
    # Sayacı sadece çizilenler için tutalım
    drawn_count = 0

    

    for prediction in object_prediction_list:
        bbox = prediction.bbox
        score = prediction.score.value
        
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        w = x2 - x1
        h = y2 - y1

        area = w*h


        # --- DEBUG BASKISI ---
        # Eğer skor 0.90'dan büyükse ama alan 1000'den küçükse (Şüpheli durum) boyutları yazdır
        if score > 0.90 and area < 1000:
            print(f"DEBUG: Şüpheli Kutu -> Skor: {score:.2f}, Boyut: {w}x{h}, Alan: {area}")

        # --- ZEKİ ÇÖPÇÜ FİLTRESİ ---
        
        # 1. ALAN FİLTRESİ: 
        # Eğer kutunun alanı 300 piksel kareden küçükse (örn: 15x20) kesin çöptür.
        #if area < 600: 
        #    continue

        # 2. KENAR ORANI (ASPECT RATIO) FİLTRESİ:
        # Profiller genelde karemsi veya hafif dikdörtgendir.
        # Eğer bir şey ipince uzunsa (örn: 100x5 piksel) o muhtemelen raf demiridir.
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 4.0 or aspect_ratio < 0.25: # Çok yatay veya çok dikeyse at
            continue

        # 3. KENAR FİLTRESİ (Opsiyonel ama etkili):
        # Eğer kutu resmin tam sınırına yapışıksa (SAHI dilimlemesinden kalan artıklar)
        height, width, _ = image.shape
        margin = 2 # 2 piksel pay
        if x1 < margin or y1 < margin or x2 > width - margin or y2 > height - margin:
            # Sadece çok küçükse ve kenardaysa at (Büyük profiller kenara değebilir)
            if area < 1000: 
                continue

        # ----


        drawn_count += 1

        # 1. KUTUYU ÇİZ
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)

        # 2. SKORU YAZ (Çok Küçük Font)
        if DRAW_TEXT:
            label = f"{score:.2f}"
            # Font ölçeği 0.3 (Çok küçük)
            font_scale = 0.3
            thickness = 1
            
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Yazı kutunun üstüne sığmıyorsa içine yaz
            text_y = y1 - 2 if y1 - t_size[1] > 2 else y1 + t_size[1] + 2
            
            # Arka plan dikdörtgeni (Yazı okunsun diye)
            cv2.rectangle(image, (x1, text_y - t_size[1] - 2), (x1 + t_size[0], text_y + 2), (0, 255, 0), -1)
            
            # Yazıyı siyah yaz (Yeşil üzerine siyah iyi okunur)
            cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    # Toplam sayıyı sol üste yaz
    count_text = f"TOPLAM: {drawn_count}"
    cv2.rectangle(image, (0, 0), (200, 40), (0, 0, 0), -1)
    cv2.putText(image, count_text, (10, 30), 0, 0.8, (0, 255, 255), 2)

    cv2.imwrite(output_path, image)
    print(f" -> Kaydedildi: {output_path} (Sayı: {drawn_count})")

def run_inference(image_path, output_folder):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov11",
        model_path=MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        device="cpu" 
    )

    image = read_image(image_path)

    #image = apply_clahe(image)

    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5
    )

    filename = os.path.basename(image_path)
    save_path = os.path.join(output_folder, f"result_{filename}")

    draw_boxes(image, result, save_path)

def apply_clahe(img):
    # Lab renk uzayına çevir (Parlaklığı renkten ayırmak için)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE uygula (Sadece L - Lightness kanalına)
    # clipLimit: Kontrast limiti (yüksekse çok sert olur, 2.0-4.0 iyidir)
    # tileGridSize: Izgara boyutu (küçük detaylar için 8x8 ideal)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Kanalları birleştir ve BGR'ye geri dön
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

def main():


    if not os.path.exists(test_folder_path):
        print("Test klasörü bulunamadı!")
        return

    os.makedirs(out_folder_path, exist_ok=True)
    
    valid_images = [f for f in os.listdir(test_folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    print(f"Toplam {len(valid_images)} resim analiz edilecek...")

    for filename in valid_images:
        full_path = os.path.join(test_folder_path, filename)
        run_inference(full_path, out_folder_path)

if __name__ == "__main__":
    main()