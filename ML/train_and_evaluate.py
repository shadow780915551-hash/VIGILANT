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

        # Always copy the image (both with and without people)
        shutil.copy(img_path, os.path.join(output_dir, "images", img_filename))
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(output_dir, "labels", label_filename)
        
        if has_person:
            # Create label file with bounding boxes
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))
        else:
            # Create empty label file for images without people
            with open(label_path, "w") as f:
                f.write("")


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
    """Train YOLOv8 model with enhanced settings for better accuracy and checkpoint support."""
    training_dir = os.path.join(ML_DIR, "training", model_name)
    last_pt_path = os.path.join(training_dir, "weights", "last.pt")
    
    # Check if we can resume from existing checkpoint
    resume_training = False
    start_epoch = 0
    
    if os.path.exists(last_pt_path):
        print(f"\n{'='*65}")
        print(f"   Found existing checkpoint: {last_pt_path}")
        print(f"   Resuming training from checkpoint...")
        print(f"{'='*65}")
        resume_training = True
        model = YOLO(last_pt_path)
    else:
        print(f"\n{'='*65}")
        print(f"   Starting fresh training with YOLOv8s")
        print(f"{'='*65}")
        # Use yolov8s (small) for better accuracy than nano, still efficient
        model = YOLO("yolov8s.pt")
    
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=os.path.join(ML_DIR, "training"),
        name=model_name,
        exist_ok=True,
        resume=resume_training,  # Resume from checkpoint if available
        plots=True,
        # Learning rate settings - more conservative for better convergence
        lr0=0.0001,           # Lower initial learning rate for stable training
        lrf=0.01,             # Final learning rate factor
        cos_lr=True,          # Cosine learning rate scheduling
        warmup_epochs=3,      # Warmup epochs for stable start
        warmup_momentum=0.8,   # Momentum during warmup
        warmup_bias_lr=0.0001,  # Learning rate for bias during warmup
        # Optimization settings
        optimizer='AdamW',    # AdamW optimizer for better performance
        momentum=0.937,       # Momentum value
        weight_decay=0.0005,  # Weight decay for regularization
        # Early stopping
        patience=20,          # Increased patience for better convergence
        save=True,            # Save checkpoints
        save_period=5,        # Save every 5 epochs (for checkpoint support)
        # Enhanced augmentation for better generalization
        augment=True,
        hsv_h=0.015,          # Hue augmentation
        hsv_s=0.7,            # Saturation augmentation
        hsv_v=0.4,            # Value augmentation
        degrees=15.0,        # Rotation degrees (increased)
        translate=0.2,        # Translation (increased)
        scale=0.6,            # Scale range (increased)
        shear=2.0,            # Shear augmentation
        fliplr=0.5,           # Horizontal flip probability
        mosaic=1.0,           # Mosaic augmentation
        mixup=0.15,           # Mixup augmentation (increased)
        copy_paste=0.15,      # Copy-paste augmentation (increased)
        # Training stability
        close_mosaic=10,      # Disable mosaic in last 10 epochs
        rect=False,           # Rectangular training disabled for better accuracy
        label_smoothing=0.1,  # Label smoothing for better generalization
        dropout=0.1,          # Dropout for regularization
        # Hardware optimization
        workers=8,            # Number of worker threads
        device=0,            # Use GPU if available
        verbose=True          # Verbose output
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
    """Evaluate model on a split using proper object detection metrics."""
    print(f"\n{'='*65}")
    print(f"   Evaluating on {split_name.upper()} set using object detection metrics...")
    print(f"{'='*65}")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Object detection metrics
    true_positives = 0
    false_positives = 0  
    false_negatives = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    
    iou_scores = []

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]

        gt_boxes = load_yolo_labels(label_path, img_width, img_height)
        total_gt_boxes += len(gt_boxes)

        # Object detection prediction
        preds = model(img, conf=conf_threshold, iou=0.45, classes=[0])[0]
        pred_boxes = preds.boxes.xyxy.cpu().numpy() if preds.boxes is not None else []
        pred_confs = preds.boxes.conf.cpu().numpy() if preds.boxes is not None else []
        
        total_pred_boxes += len(pred_boxes)

        # Match predictions to ground truth using IoU
        used_gt = set()
        used_pred = set()
        
        for i, pred_box in enumerate(pred_boxes):
            if i in used_pred:
                continue
            best_iou, best_gt_idx = 0, -1
            for j, gt_box in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, j
            
            if best_iou >= 0.5:  # IoU threshold for true positive
                true_positives += 1
                used_gt.add(best_gt_idx)
                used_pred.add(i)
                iou_scores.append(best_iou)
            else:
                false_positives += 1

        # Count unmatched ground truth boxes as false negatives
        for j in range(len(gt_boxes)):
            if j not in used_gt:
                false_negatives += 1

    # Calculate object detection metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(iou_scores) if len(iou_scores) > 0 else 0

    # Print results
    print(f"\n{'='*65}")
    print(f"   {split_name.upper()} SET - {model_name} OBJECT DETECTION RESULTS")
    print(f"{'='*65}")
    print(f"   Total Ground Truth Boxes:   {total_gt_boxes}")
    print(f"   Total Predicted Boxes:     {total_pred_boxes}")
    print(f"   True Positives:            {true_positives}")
    print(f"   False Positives:           {false_positives}")
    print(f"   False Negatives:           {false_negatives}")
    print(f"{'='*65}")
    print(f"   Precision:                 {precision:.4f}")
    print(f"   Recall:                    {recall:.4f}")
    print(f"   F1-Score:                  {f1:.4f}")
    print(f"   Average IoU:               {avg_iou:.4f}")
    print(f"{'='*65}\n")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "avg_iou": float(avg_iou),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "total_gt_boxes": int(total_gt_boxes),
        "total_pred_boxes": int(total_pred_boxes)
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

    # Train or load model (50 epochs for better accuracy)
    model_name = "yolov8s_50epochs_enhanced"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")

    if os.path.exists(model_path):
        print(f"\nModel {model_name} already exists! Loading pre-trained model...")
        model = YOLO(model_path)
    else:
        print(f"\nTraining enhanced model for 50 epochs...")
        print("Using YOLOv8s with optimized hyperparameters for better accuracy")
        model = train_model(yaml_path, 50, model_name)

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
    print("\n✅ Training and evaluation completed successfully!")
    print("   Note: Using object detection metrics (not binary classification)")
    print("   This is the correct approach for person detection tasks")


if __name__ == "__main__":
    main()