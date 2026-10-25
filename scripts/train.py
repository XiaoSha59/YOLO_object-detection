import os
import shutil
import argparse
from pathlib import Path
import yaml
import zipfile
from ultralytics import YOLO

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

    print(f"[BACKUP] Copied artifacts {copied} for {model_name} to {dest_dir}")

    zip_path = Path("backup.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(backup_root):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(backup_root.parent)
                zipf.write(file_path, arcname)
    print(f"[BACKUP] Updated backup archive: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.1f} KB)")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 BCCD v3 Training Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test with YOLO26m to verify VRAM")
    parser.add_argument("--model", type=str, default="all", choices=["all", "yolo26n", "yolo26s", "yolo26m"], help="Select specific model to train")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt checkpoint")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size from train.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--device", default=None, help="Device ID (e.g. 0 or cpu)")
    args = parser.parse_args()

    config_path = Path("configs/train.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = load_yaml(config_path)

    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    # Smoke test mode
    if args.smoke_test:
        print("=" * 65)
        print("[SMOKE TEST] Initializing YOLO26m (1 epoch VRAM check)")
        print("=" * 65)
        smoke_model = YOLO("yolo26m.pt")
        smoke_model.train(
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
        print("[SUCCESS] Smoke test completed successfully. Ready for full training.")
        return

    all_models = [
        ("yolo26n", "yolo26n.pt"),
        ("yolo26s", "yolo26s.pt"),
        ("yolo26m", "yolo26m.pt")
    ]

    if args.model != "all":
        models_to_train = [m for m in all_models if m[0] == args.model]
    else:
        models_to_train = all_models

    print("=" * 65)
    print(f"[TRAINING] Target models: {[m[0] for m in models_to_train]}")
    print(f"[TRAINING] Config: epochs={cfg['epochs']}, batch={cfg['batch']}, imgsz={cfg['imgsz']}, optimizer={cfg['optimizer']}")
    print("=" * 65)

    for tag, weight_name in models_to_train:
        print(f"\n[INFO] Starting training for: {tag}...")
        run_project = "runs/detect"
        last_ckpt = Path(run_project) / tag / "weights" / "last.pt"

        if args.resume and last_ckpt.exists():
            print(f"[INFO] Found interrupted checkpoint at {last_ckpt}. Resuming training...")
            model = YOLO(str(last_ckpt))
            model.train(resume=True)
        else:
            try:
                model = YOLO(weight_name)
            except Exception:
                fallback_name = weight_name.replace("yolo26", "yolov26")
                print(f"[WARNING] Weight {weight_name} not found. Trying fallback {fallback_name}...")
                model = YOLO(fallback_name)

            model.train(
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

    print("\n[SUCCESS] Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
