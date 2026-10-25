import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml
import zipfile
from ultralytics import YOLO

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def backup_model_run(run_dir, model_name, backup_root=Path("backup")):
    backup_root.mkdir(parents=True, exist_ok=True)
    dest_dir = backup_root / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "results.png"]
    copied = []
    for f in files_to_copy:
        src = run_dir / f
        if src.exists():
            dst = dest_dir / Path(f).name
            shutil.copy2(src, dst)
            copied.append(Path(f).name)

    print(f"📦 Đã sao lưu {copied} của {model_name} vào: {dest_dir}")

    # Tạo file zip tổng hợp
    zip_path = Path("backup.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(backup_root):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(backup_root.parent)
                zipf.write(file_path, arcname)
    print(f"🗜️ Đã cập nhật file nén sao lưu: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.1f} KB)")

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO26 trên BCCD v3")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy thử nghiệm 1 epoch với YOLO26m để kiểm tra VRAM")
    parser.add_argument("--batch", type=int, default=None, help="Ghi đè batch size (mặc định theo train.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epochs")
    parser.add_argument("--device", default=None, help="Device (0 hoặc cpu)")
    args = parser.parse_args()

    config_path = Path("configs/train.yaml")
    if not config_path.exists():
        raise FileNotFoundError("Không tìm thấy file configs/train.yaml!")

    cfg = load_yaml(config_path)

    # Override nếu có tham số từ dòng lệnh
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    # Smoke test mode
    if args.smoke_test:
        print("\n" + "=" * 60)
        print("🚀 BẮT ĐẦU SMOKE TEST: YOLO26m (1 epoch)")
        print("=" * 60)
        smoke_model = YOLO("yolo26m.pt")
        smoke_res = smoke_model.train(
            data=cfg["data"],
            epochs=1,
            imgsz=cfg.get("imgsz", 640),
            batch=cfg.get("batch", 8),
            device=cfg.get("device", 0),
            project="runs/smoke_test",
            name="yolo26m_smoke",
            exist_ok=True,
            amp=cfg.get("amp", True)
        )
        print("\n✅ SMOKE TEST THÀNH CÔNG! Sẵn sàng train toàn bộ 3 model.")
        return

    # Full training pipeline: n -> s -> m
    models_to_train = [
        ("yolo26n", "yolo26n.pt"),
        ("yolo26s", "yolo26s.pt"),
        ("yolo26m", "yolo26m.pt")
    ]

    print("\n" + "=" * 60)
    print("🎯 BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC: YOLO26n -> YOLO26s -> YOLO26m")
    print(f"⚙️ Config: epochs={cfg['epochs']}, batch={cfg['batch']}, imgsz={cfg['imgsz']}, optimizer={cfg['optimizer']}")
    print("=" * 60)

    for tag, weight_name in models_to_train:
        print(f"\n▶️ Đang huấn luyện mô hình: {tag} ({weight_name})...")
        try:
            model = YOLO(weight_name)
        except Exception as e:
            # Fallback nếu tên checkpoint có dạng yolov26n.pt
            fallback_name = weight_name.replace("yolo26", "yolov26")
            print(f"⚠️ Thử tải fallback: {fallback_name}...")
            model = YOLO(fallback_name)

        run_project = "runs/detect"
        results = model.train(
            data=cfg["data"],
            epochs=cfg["epochs"],
            patience=cfg.get("patience", 10),
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            optimizer=cfg.get("optimizer", "AdamW"),
            lr0=cfg.get("lr0", 0.001),
            weight_decay=cfg.get("weight_decay", 0.0005),
            seed=cfg.get("seed", 42),
            deterministic=cfg.get("deterministic", True),
            amp=cfg.get("amp", True),
            workers=cfg.get("workers", 2),
            close_mosaic=cfg.get("close_mosaic", 5),
            cache=cfg.get("cache", "ram"),
            device=cfg.get("device", 0),
            project=run_project,
            name=tag,
            exist_ok=True
        )

        run_dir = Path(run_project) / tag
        backup_model_run(run_dir, tag)

    print("\n🎉 HOÀN TẤT HUẤN LUYỆN TOÀN BỘ 3 MÔ HÌNH!")

if __name__ == "__main__":
    main()
