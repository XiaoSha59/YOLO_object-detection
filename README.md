# YOLO26 Blood Cell Detection & Counting (BCCD v3)

A complete pipeline for detecting and counting blood cells (Platelets, RBC, WBC) on the **BCCD v3** dataset using the **Ultralytics YOLO26** model family (`yolo26n`, `yolo26s`, `yolo26m`).

---

## 📁 Directory Structure

```text
YOLO_object-detection/
├── configs/
│   └── train.yaml          # Locked hyperparameters (AdamW, batch 8, imgsz 640)
├── datasets/
│   └── bccd/               # BCCD v3 dataset (255 train / 73 valid / 36 test)
│       ├── data.yaml       # Dataset config (Platelets, RBC, WBC)
│       ├── train/          # images/ and labels/
│       ├── valid/          # images/ and labels/
│       └── test/           # images/ and labels/
├── figures/                # Visualizations and sample predictions
├── metrics/                # CSV outputs from evaluation runs
├── report/
│   └── report.md           # Experiment analysis and performance report
├── scripts/
│   ├── download_data.py    # Fetch BCCD v3 via Roboflow API
│   ├── check_dataset.py    # Validate split counts and box integrity
│   ├── train.py            # Sequential training (n -> s -> m) & auto-backup
│   ├── eval_test.py        # Benchmark on test set (conf=0.001)
│   └── demo_count.py       # Cell counting visualization (conf=0.25)
├── .gitignore              # Ignores .venv, .env, *.pt, *plan.md
├── requirements.txt        # Pinned dependencies (ultralytics>=8.4.0)
└── README.md
```

---

## ⚙️ Initial Setup

### 1. Environment & Dependencies
Clone the repository and set up a virtual environment:

```bash
# Clone repository
git clone https://github.com/XiaoSha59/YOLO_object-detection.git
cd YOLO_object-detection

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
# source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Setup (BCCD v3)

The dataset is partitioned into official fixed splits:
* **Train**: 255 images (3,450 boxes)
* **Valid**: 73 images (967 boxes)
* **Test**: 36 images (471 boxes)
* **Total**: 364 images across 3 classes:
  * `0`: **Platelets** (361 boxes)
  * `1`: **RBC** (4,155 boxes)
  * `2`: **WBC** (372 boxes)

To verify dataset integrity and generate a 12-sample preview (`figures/dataset_samples.png`):
```bash
python scripts/check_dataset.py
```

---

## 🚀 Ultralytics YOLO26 Workflow

YOLO26 introduces native end-to-end NMS-free inference and removes Distribution Focal Loss (DFL) for improved edge and GPU efficiency.

### 1. Smoke Test (VRAM & Pipeline Check)
Run a quick 1-epoch smoke test with `yolo26m` before starting full training:
```bash
python scripts/train.py --smoke-test
```

### 2. Full Training (Sequential $n \rightarrow s \rightarrow m$)
Train all three models sequentially with locked hyperparameters (`AdamW`, $lr_0=0.001$, `batch=8`, `epochs=40`):
```bash
python scripts/train.py
```
> **Note**: After each model finishes, weights (`best.pt`), training curves, and `results.csv` are automatically archived into `backup/` and compressed to `backup.zip`.

### 3. Test Set Evaluation
Evaluate all trained models on the test set (`conf=0.001`):
```bash
python scripts/eval_test.py
```
Outputs precision, recall, mAP50, mAP50-95, and class APs to `metrics/comparison_test.csv`.

### 4. Cell Counting Demo
Run detection and cell counting on test images (`conf=0.25`):
```bash
python scripts/demo_count.py
```
Saves annotated cell counts to `figures/demo_counting.png`.
