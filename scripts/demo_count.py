import sys
import os
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Demo đếm tế bào máu trên ảnh test với YOLO26")
    parser.add_argument("--image", default=None, help="Đường dẫn file ảnh (mặc định lấy ảnh đầu tiên trong test/images)")
    parser.add_argument("--model", default=None, help="Đường dẫn checkpoint best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence (mặc định: 0.25)")
    args = parser.parse_args()

    # Tìm model
    model_path = args.model
    if not model_path:
        for cand in [
            "backup/yolo26s/best.pt",
            "runs/detect/yolo26s/weights/best.pt",
            "backup/yolo26m/best.pt",
            "runs/detect/yolo26m/weights/best.pt",
            "backup/yolo26n/best.pt",
            "runs/detect/yolo26n/weights/best.pt",
        ]:
            if Path(cand).exists():
                model_path = cand
                break

    if not model_path or not Path(model_path).exists():
        print(f"⚠️ Chưa tìm thấy checkpoint best.pt! Hãy huấn luyện xong hoặc chỉ định qua --model.")
        return

    # Tìm ảnh
    image_path = args.image
    if not image_path:
        test_images = sorted(list(Path("datasets/bccd/test/images").glob("*.jpg")) + list(Path("datasets/bccd/test/images").glob("*.png")))
        if not test_images:
            print("❌ Không tìm thấy ảnh trong datasets/bccd/test/images!")
            return
        image_path = str(test_images[0])

    print(f"🔬 Đang chạy demo đếm tế bào với mô hình: {model_path}")
    print(f"🖼️ Ảnh đầu vào: {image_path} (conf={args.conf})")

    model = YOLO(model_path)
    results = model.predict(image_path, conf=args.conf, save=False)
    res = results[0]

    class_names = {0: "Platelets", 1: "RBC", 2: "WBC"}
    counts = {0: 0, 1: 0, 2: 0}

    boxes = res.boxes
    if boxes is not None:
        for cls_id in boxes.cls.cpu().numpy():
            cls_int = int(cls_id)
            counts[cls_int] = counts.get(cls_int, 0) + 1

    print("\n" + "=" * 50)
    print("🩸 KẾT QUẢ ĐẾM TẾ BÀO MÁU:")
    print(f"   - Hồng cầu (RBC):       {counts.get(1, 0)} tế bào")
    print(f"   - Bạch cầu (WBC):       {counts.get(2, 0)} tế bào")
    print(f"   - Tiểu cầu (Platelets): {counts.get(0, 0)} tế bào")
    print(f"   👉 TỔNG CỘNG:           {sum(counts.values())} tế bào")
    print("=" * 50)

    # Vẽ và lưu ảnh kết quả
    annotated_img = res.plot() # BGR image from Ultralytics
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "demo_counting.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_rgb)
    title_text = f"BCCD Blood Cell Counting (YOLO26)\nRBC: {counts.get(1, 0)} | WBC: {counts.get(2, 0)} | Platelets: {counts.get(0, 0)}"
    plt.title(title_text, fontsize=13, fontweight='bold', pad=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\n✅ Đã lưu ảnh trực quan kết quả đếm vào: {out_path}")

if __name__ == "__main__":
    main()
