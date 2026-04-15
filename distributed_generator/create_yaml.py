"""3 script bittikten sonra çalıştır - data.yaml oluşturur ve istatistik verir."""
import os

BASE_DIR = "dataset_5_profile_FINAL"
CLASS_ORDER = ['1524', '1560', '9859', '7170', '9794']

def main():
    with open(f'{BASE_DIR}/data.yaml', 'w') as f:
        names = "\n".join([f"  {i}: '{n}'" for i, n in enumerate(CLASS_ORDER)])
        f.write(f"path: {os.path.abspath(BASE_DIR)}\ntrain: images/train\nval: images/val\nnames:\n{names}\nnc: {len(CLASS_ORDER)}")
    print(f"\n📊 Dataset İstatistikleri:")
    for subset in ['train', 'val']:
        imgs = len([f for f in os.listdir(f'{BASE_DIR}/images/{subset}') if f.endswith('.jpg')])
        lbls = len([f for f in os.listdir(f'{BASE_DIR}/labels/{subset}') if f.endswith('.txt')])
        print(f"   {subset}: {imgs} resim, {lbls} etiket")
    for cls in CLASS_ORDER:
        t = len([f for f in os.listdir(f'{BASE_DIR}/images/train') if f.startswith(cls)])
        v = len([f for f in os.listdir(f'{BASE_DIR}/images/val') if f.startswith(cls)])
        print(f"   {cls}: Train={t}, Val={v}")
    print("✅ data.yaml oluşturuldu!")

if __name__ == "__main__":
    main()
