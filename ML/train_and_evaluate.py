import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from ultralytics import YOLO
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ML_DIR = os.path.join(BASE_DIR, "ML")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def coco_to_yolo(coco_json_path, images_dir, output_dir):
    """Convert COCO annotations to YOLO format (person class only)."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    categories = coco_data["categories"]
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
    person_idx = None
    for idx, cat in enumerate(categories):
        if cat["name"] == "person":
            person_idx = idx
            break

    if person_idx is None:
        raise ValueError("Person category not found in COCO annotations")

    images = coco_data["images"]
    annotations = coco_data["annotations"]

    image_id_to_info = {img["id"]: img for img in images}
    image_id_to_anns = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)

    for img_id, img_info in image_id_to_info.items():
        img_filename = img_info["file_name"]
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            continue

        img_height = img_info["height"]
        img_width = img_info["width"]

        anns = image_id_to_anns.get(img_id, [])
        label_lines = []
        has_person = False

        for ann in anns:
            if ann["category_id"] not in cat_id_to_idx:
                continue
            if cat_id_to_idx[ann["category_id"]] != person_idx:
                continue

            has_person = True
            bbox = ann["bbox"]
            x, y, w, h = bbox
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height

            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if has_person:
            shutil.copy(img_path, os.path.join(output_dir, "images", img_filename))
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            label_path = os.path.join(output_dir, "labels", label_filename)
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))


def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Split real COCO images into train/val/test (no synthetic data)."""
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    splits = {
        "train": (train_ratio, 0),
        "val": (val_ratio, train_ratio),
        "test": (1.0 - train_ratio - val_ratio, train_ratio + val_ratio)
    }

    for split_name in splits:
        os.makedirs(os.path.join(output_dir, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split_name, "labels"), exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    np.random.seed(42)
    np.random.shuffle(image_files)
    n = len(image_files)

    for split_name, (ratio, offset) in splits.items():
        start = int(n * offset)
        end = int(n * (offset + ratio))
        split_files = image_files[start:end]
        img_dest = os.path.join(output_dir, split_name, "images")
        lbl_dest = os.path.join(output_dir, split_name, "labels")
        for f in split_files:
            shutil.copy(os.path.join(images_dir, f), os.path.join(img_dest, f))
            label_f = os.path.splitext(f)[0] + ".txt"
            shutil.copy(os.path.join(labels_dir, label_f), os.path.join(lbl_dest, label_f))
        print(f"   {split_name}: {len(split_files)} images")


def create_yaml_config(dataset_dir, output_path):
    """Create YOLO dataset config YAML."""
    config_content = f"""path: {dataset_dir}
train: train/images
val: val/images
test: test/images
names:
  0: person
"""
    with open(output_path, "w") as f:
        f.write(config_content)


def train_model(yaml_path, epochs, model_name):
    """Train YOLOv8n model with best settings for accuracy."""
    model = YOLO("yolov8n.pt")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        project=os.path.join(ML_DIR, "training"),
        name=model_name,
        exist_ok=True,
        plots=True,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=15,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1
    )
    best_model_path = os.path.join(ML_DIR, "training", model_name, "weights", "best.pt")
    shutil.copy(best_model_path, os.path.join(MODELS_DIR, f"{model_name}.pt"))
    return YOLO(os.path.join(MODELS_DIR, f"{model_name}.pt"))


def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO format labels -> [x1,y1,x2,y2] boxes."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, xc, yc, w, h = map(float, line.strip().split())
                if class_id == 0:
                    x1 = (xc - w/2) * img_width
                    y1 = (yc - h/2) * img_height
                    x2 = (xc + w/2) * img_width
                    y2 = (yc + h/2) * img_height
                    boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def evaluate_on_split(model, split_name, images_dir, labels_dir, model_name, conf_threshold=0.5):
    """Evaluate model on a split and compute all metrics + plots."""
    print(f"\n{'='*65}")
    print(f"   Evaluating on {split_name.upper()} set...")
    print(f"{'='*65}")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    y_true_binary, y_scores = [], []
    y_true_class, y_pred_class = [], []

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]

        gt_boxes = load_yolo_labels(label_path, img_width, img_height)
        has_gt_person = len(gt_boxes) > 0

        preds = model(img, conf=0.001, classes=[0])[0]
        pred_boxes = preds.boxes.xyxy.cpu().numpy() if preds.boxes is not None else []
        pred_confs = preds.boxes.conf.cpu().numpy() if preds.boxes is not None else []

        has_pred_person = len(pred_boxes) > 0 and max(pred_confs) > conf_threshold if len(pred_confs) > 0 else False
        y_true_class.append(1 if has_gt_person else 0)
        y_pred_class.append(1 if has_pred_person else 0)

        used_gt = set()
        for i, pred_box in enumerate(pred_boxes):
            best_iou, best_gt_idx = 0, -1
            for j, gt_box in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, j
            if best_iou >= 0.5:
                y_true_binary.append(1)
                used_gt.add(best_gt_idx)
            else:
                y_true_binary.append(0)
            y_scores.append(pred_confs[i])

        for j in range(len(gt_boxes)):
            if j not in used_gt:
                y_true_binary.append(1)
                y_scores.append(0.0)

    # Compute metrics
    cm = confusion_matrix(y_true_class, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)# Print results
    print(f"\n{'='*65}")
    print(f"   {split_name.upper()} SET - {model_name} RESULTS")
    print(f"{'='*65}")
    print(f"   Accuracy:              {accuracy:.4f}")
    print(f"   Precision:             {precision:.4f}")
    print(f"   Recall:                {recall:.4f}")
    print(f"   F1-Score:              {f1:.4f}")
    print(f"   AUC:                   {roc_auc:.4f}")
    print(f"{'='*65}")
    print(f"   Confusion Matrix:")
    print(f"   TN: {tn}  FP: {fp}")
    print(f"   FN: {fn}  TP: {tp}")
    print(f"{'='*65}\n")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Person', 'Person'],
                yticklabels=['No Person', 'Person'],
                annot_kws={"size": 14})
    plt.title(f'Confusion Matrix - {split_name.upper()} Set', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    cm_path = os.path.join(ML_DIR, f"confusion_matrix_{model_name}_{split_name}.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"   Confusion Matrix saved: {cm_path}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#0066cc', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#999999', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {split_name.upper()} Set', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    roc_path = os.path.join(ML_DIR, f"roc_curve_{model_name}_{split_name}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ROC Curve saved:        {roc_path}")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }


def main():
    yaml_path = os.path.join(ML_DIR, "dataset.yaml")
    yolo_dataset_dir = os.path.join(ML_DIR, "yolo_dataset")

    # Prepare dataset if not already done
    if not os.path.exists(yaml_path) or not os.path.exists(os.path.join(yolo_dataset_dir, "train", "images")):
        print("Step 1: Converting COCO annotations to YOLO format...")
        coco_val_json = os.path.join(DATASET_DIR, "annotations", "annotations", "instances_val2017.json")
        val_images_dir = os.path.join(DATASET_DIR, "val2017", "val2017")
        yolo_temp_dir = os.path.join(ML_DIR, "yolo_temp")
        coco_to_yolo(coco_val_json, val_images_dir, yolo_temp_dir)

        print("Step 2: Splitting dataset into train/val/test...")
        split_dataset(yolo_temp_dir, yolo_dataset_dir)

        print("Step 3: Creating YOLO dataset config...")
        create_yaml_config(yolo_dataset_dir, yaml_path)
    else:
        print("Dataset already prepared! Skipping data preparation steps.")

    # Train or load model (25 epochs)
    model_name = "yolov8n_25epochs"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")

    if os.path.exists(model_path):
        print(f"\nModel {model_name} already exists! Loading pre-trained model...")
        model = YOLO(model_path)
    else:
        print(f"\nTraining model for 25 epochs...")
        model = train_model(yaml_path, 25, model_name)

    # Run YOLO's built-in validation (for mAP metrics)
    print("\nRunning YOLO validation (mAP metrics)...")
    val_results = model.val(data=yaml_path, plots=True)
    print(f"\nYOLO Validation Metrics:")
    print(f"   mAP50:    {val_results.box.map50:.4f}")
    print(f"   mAP50-95: {val_results.box.map:.4f}")
    print(f"   Precision:{val_results.box.mp:.4f}")
    print(f"   Recall:   {val_results.box.mr:.4f}")

    # Evaluate on Validation set
    val_images_dir = os.path.join(yolo_dataset_dir, "val", "images")
    val_labels_dir = os.path.join(yolo_dataset_dir, "val", "labels")
    val_metrics = evaluate_on_split(model, "val", val_images_dir, val_labels_dir, model_name)

    # Evaluate on Test set
    test_images_dir = os.path.join(yolo_dataset_dir, "test", "images")
    test_labels_dir = os.path.join(yolo_dataset_dir, "test", "labels")
    test_metrics = evaluate_on_split(model, "test", test_images_dir, test_labels_dir, model_name)

    # Save all metrics
    all_metrics = {
        "model": model_name,
        "yolo_val_metrics": {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr)
        },
        "val_set": val_metrics,
        "test_set": test_metrics
    }
    metrics_file = os.path.join(ML_DIR, f"{model_name}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*65}")
    print(f"   All metrics saved to: {metrics_file}")
    print(f"{'='*65}")
    print("\n✅ All steps completed successfully!")


if __name__ == "__main__":
    main()