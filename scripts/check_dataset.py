import sys
import glob
from pathlib import Path
import yaml
import cv2
import matplotlib.pyplot as plt

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def check_dataset():
    data_yaml_path = Path("datasets/bccd/data.yaml")
    if not data_yaml_path.exists():
        print(" Error: datasets/bccd/data.yaml not found!")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_dir = Path("datasets/bccd")
    splits = {
        "train": base_dir / "train",
        "valid": base_dir / "valid",
        "test": base_dir / "test"
    }

    class_names = data_cfg.get("names", {0: "Platelets", 1: "RBC", 2: "WBC"})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    print("=" * 60)
    print("🔍 KIỂM TRA TÍNH TOÀN VẸN CỦA TẬP DỮ LIỆU BCCD v3")
    print("=" * 60)

    total_images = 0
    all_image_names = {}
    class_box_counts = {cls_id: 0 for cls_id in class_names.keys()}
    all_valid = True

    for split_name, split_path in splits.items():
        img_dir = split_path / "images"
        lbl_dir = split_path / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"❌ Thiếu thư mục: {img_dir} hoặc {lbl_dir}")
            all_valid = False
            continue

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        labels = sorted(list(lbl_dir.glob("*.txt")))

        num_imgs = len(images)
        num_lbls = len(labels)
        total_images += num_imgs

        print(f"\n📂 Split: [{split_name.upper()}]")
        print(f"   - Số lượng ảnh:  {num_imgs}")
        print(f"   - Số lượng nhãn: {num_lbls}")

        img_stems = {img.stem for img in images}
        lbl_stems = {lbl.stem for lbl in labels}

        # Check pairing
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems

        if missing_labels:
            print(f"   ⚠️ Có {len(missing_labels)} ảnh không có file nhãn tương ứng!")
            all_valid = False
        if missing_images:
            print(f"   ⚠️ Có {len(missing_images)} nhãn không có ảnh tương ứng!")
            all_valid = False

        # Check duplicates across splits
        for img_name in img_stems:
            if img_name in all_image_names:
                print(f"   ⚠️ Trùng tên ảnh '{img_name}' giữa split '{split_name}' và '{all_image_names[img_name]}'")
                all_valid = False
            else:
                all_image_names[img_name] = split_name

        # Validate box bounds & class counts
        split_box_count = 0
        invalid_boxes = 0
        for lbl_file in labels:
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])

                    split_box_count += 1
                    class_box_counts[cls_id] = class_box_counts.get(cls_id, 0) + 1

                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        invalid_boxes += 1

        print(f"   - Tổng số bounding boxes: {split_box_count}")
        if invalid_boxes > 0:
            print(f"   ❌ Có {invalid_boxes} box có tọa độ ngoài khoảng [0, 1]!")
            all_valid = False
        else:
            print(f"   ✅ Tất cả box đều hợp lệ trong khoảng [0, 1].")

    print("\n" + "=" * 60)
    print("📊 THỐNG KÊ SỐ LƯỢNG BOX THEO CLASS TOÀN BỘ DATASET:")
    for cls_id, count in sorted(class_box_counts.items()):
        print(f"   - Class {cls_id} ({class_names.get(cls_id, 'Unknown')}): {count} boxes")
    print(f"   - Tổng số ảnh toàn dataset: {total_images} (Kỳ vọng: 255 + 73 + 36 = 364)")
    print("=" * 60)

    # Visualize 12 sample images with ground truth boxes
    visualize_samples(splits["train"], class_names, num_samples=12)

    return all_valid

def visualize_samples(split_path, class_names, num_samples=12):
    img_dir = split_path / "images"
    lbl_dir = split_path / "labels"
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:num_samples]

    if not images:
        return

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_samples.png"

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)} # Platelets, RBC, WBC

    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img.shape

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        x1 = int((x_c - w / 2) * w_img)
                        y1 = int((y_c - h / 2) * h_img)
                        x2 = int((x_c + w / 2) * w_img)
                        y2 = int((y_c + h / 2) * h_img)

                        color = colors.get(cls_id, (255, 255, 0))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        axes[idx].imshow(img)
        axes[idx].set_title(img_path.name, fontsize=9)
        axes[idx].axis("off")

    plt.suptitle("BCCD v3 Ground Truth Samples (12 Images)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n🖼️ Đã lưu hình ảnh trực quan 12 mẫu vào: {output_path}")

if __name__ == "__main__":
    check_dataset()
