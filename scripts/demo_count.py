import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Blood cell detection and counting demo on BCCD test set")
    parser.add_argument("--image", default=None, help="Path to input test image")
    parser.add_argument("--model", default=None, help="Path to best.pt checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        cand_list = [
            "runs/detect/yolo26s/weights/best.pt",
            "runs/detect/runs/detect/yolo26s/weights/best.pt",
            "backup/yolo26s/best.pt",
            "runs/detect/yolo26m/weights/best.pt",
            "runs/detect/runs/detect/yolo26m/weights/best.pt",
            "backup/yolo26m/best.pt",
            "runs/detect/yolo26n/weights/best.pt",
            "runs/detect/runs/detect/yolo26n/weights/best.pt",
            "backup/yolo26n/best.pt",
        ] + [str(p) for p in Path("runs").rglob("best.pt")] + [str(p) for p in Path("backup").rglob("best.pt")]
        for cand in cand_list:
            if Path(cand).exists():
                model_path = cand
                break

    if not model_path or not Path(model_path).exists():
        print("[WARNING] Checkpoint best.pt not found. Please provide path via --model.")
        return

    image_path = args.image
    if not image_path:
        test_images = sorted(list(Path("datasets/bccd/test/images").glob("*.jpg")) + list(Path("datasets/bccd/test/images").glob("*.png")))
        if not test_images:
            print("[ERROR] No images found in datasets/bccd/test/images.")
            return
        image_path = str(test_images[0])

    print(f"[INFO] Running cell counting inference with model: {model_path}")
    print(f"[INFO] Input image: {image_path} (confidence threshold: {args.conf})")

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
    print("CELL COUNTING SUMMARY:")
    print(f"  - Red Blood Cells (RBC):     {counts.get(1, 0)}")
    print(f"  - White Blood Cells (WBC):   {counts.get(2, 0)}")
    print(f"  - Platelets:                 {counts.get(0, 0)}")
    print(f"  - TOTAL DETECTED CELLS:      {sum(counts.values())}")
    print("=" * 50)

    annotated_img = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "demo_counting.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_rgb)
    title_text = f"BCCD Blood Cell Counting (YOLO26)\nRBC: {counts.get(1, 0)} | WBC: {counts.get(2, 0)} | Platelets: {counts.get(0, 0)}"
    plt.title(title_text, fontsize=12, fontweight='bold', pad=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[SUCCESS] Visualized prediction saved to: {out_path}")

if __name__ == "__main__":
    main()
