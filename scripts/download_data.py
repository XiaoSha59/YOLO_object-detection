import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

def download_bccd():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not found in .env file.")

    print("[INFO] Connecting to Roboflow API...")
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("joseph-nelson").project("bccd")
    version = project.version(3)

    target_dir = Path("datasets/bccd").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading BCCD v3 dataset (format: yolov8) to {target_dir}...")
    version.download("yolov8", location=str(target_dir))

    data_yaml_path = target_dir / "data.yaml"
    if data_yaml_path.exists():
        print(f"[INFO] Normalizing dataset paths in {data_yaml_path}...")
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data["path"] = "datasets/bccd"
        data["train"] = "train/images"
        data["val"] = "valid/images"
        data["test"] = "test/images"

        with open(data_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print("[INFO] Successfully updated datasets/bccd/data.yaml.")

    print("[SUCCESS] Dataset download and configuration completed successfully.")

if __name__ == "__main__":
    download_bccd()
