import sys
import os
from pathlib import Path
import pandas as pd
import yaml
from ultralytics import YOLO

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def evaluate_models():
    data_yaml = "datasets/bccd/data.yaml"
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models = ["yolo26n", "yolo26s", "yolo26m"]
    rows = []

    print("=" * 65)
    print("📋 ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TẬP TEST (BCCD v3, conf=0.001)")
    print("=" * 65)

    for model_tag in models:
        # Check either runs/detect/model_tag/weights/best.pt or backup/model_tag/best.pt
        cand_paths = [
            Path(f"runs/detect/{model_tag}/weights/best.pt"),
            Path(f"backup/{model_tag}/best.pt"),
            Path(f"checkpoints/{model_tag}/best.pt"),
        ]
        best_pt = None
        for p in cand_paths:
            if p.exists():
                best_pt = p
                break

        if not best_pt:
            print(f"⚠️ Chưa tìm thấy checkpoint cho {model_tag}. Bỏ qua.")
            continue

        file_size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\n🔬 Đang đánh giá {model_tag} ({best_pt})...")
        model = YOLO(str(best_pt))

        # Run validation on test split
        metrics = model.val(
            data=data_yaml,
            split="test",
            conf=0.001,
            device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            project="runs/eval_test",
            name=model_tag,
            exist_ok=True
        )

        # Trích xuất số liệu
        box = metrics.box
        p = float(box.mp)
        r = float(box.mr)
        map50 = float(box.map50)
        map50_95 = float(box.map)

        # AP từng class
        class_maps = box.maps
        # Names: 0: Platelets, 1: RBC, 2: WBC
        ap_platelets = float(class_maps[0]) if len(class_maps) > 0 else 0.0
        ap_rbc = float(class_maps[1]) if len(class_maps) > 1 else 0.0
        ap_wbc = float(class_maps[2]) if len(class_maps) > 2 else 0.0

        rows.append({
            "Model": model_tag,
            "Size (MB)": round(file_size_mb, 2),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
            "AP50_Platelets": round(ap_platelets, 4),
            "AP50_RBC": round(ap_rbc, 4),
            "AP50_WBC": round(ap_wbc, 4),
        })

    if rows:
        df = pd.DataFrame(rows)
        out_csv = metrics_dir / "comparison_test.csv"
        df.to_csv(out_csv, index=False)
        print("\n" + "=" * 65)
        print(f"📊 KẾT QUẢ SO SÁNH TRÊN TEST SET (Đã lưu tại {out_csv}):")
        print("=" * 65)
        print(df.to_string(index=False))
    else:
        print("⚠️ Chưa có checkpoint nào được đánh giá!")

if __name__ == "__main__":
    evaluate_models()
