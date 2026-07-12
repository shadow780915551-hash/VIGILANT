
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from ultralytics import YOLO
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ML_DIR = os.path.join(BASE_DIR, "ML")

os.makedirs(ML_DIR, exist_ok=True)


def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO format labels and convert to [x1,y1,x2,y2]"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, xc, yc, w, h = map(float, line.strip().split())
                if class_id == 0:  # Only person class
                    x1 = (xc - w/2) * img_width
                    y1 = (yc - h/2) * img_height
                    x2 = (xc + w/2) * img_width
                    y2 = (yc + h/2) * img_height
                    boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def main():
    print("🚀 Evaluating Pre-Trained YOLOv8n on COCO Val2017...")

    # Load pre-trained model
    model = YOLO("yolov8n.pt")

    # Check if dataset is prepared
    yaml_path = os.path.join(ML_DIR, "dataset.yaml")
    yolo_dataset_dir = os.path.join(ML_DIR, "yolo_dataset")
    val_images_dir = os.path.join(yolo_dataset_dir, "val", "images")
    val_labels_dir = os.path.join(yolo_dataset_dir, "val", "labels")
    
    if not os.path.exists(yaml_path) or not os.path.exists(val_images_dir):
        print("⚠️ Dataset not prepared yet! Let's prepare it first...")
        from train_and_evaluate import coco_to_yolo, split_dataset, create_yaml_config
        coco_val_json = os.path.join(DATASET_DIR, "annotations", "annotations", "instances_val2017.json")
        val_images_dir_src = os.path.join(DATASET_DIR, "val2017", "val2017")
        yolo_temp_dir = os.path.join(ML_DIR, "yolo_temp")
        coco_to_yolo(coco_val_json, val_images_dir_src, yolo_temp_dir)
        split_dataset(yolo_temp_dir, yolo_dataset_dir)
        create_yaml_config(yolo_dataset_dir, yaml_path)
        print("✅ Dataset prepared!")

    # First run YOLO's val to get standard metrics and plots
    results = model.val(data=yaml_path, plots=True)

    # Now collect predictions for ROC curve and confusion matrix
    print("\n🔍 Collecting predictions for ROC and Confusion Matrix...")
    y_true_binary = []  # For ROC (0: no person, 1: person)
    y_scores = []       # Confidence scores
    y_true_class = []   # For confusion matrix (image-level)
    y_pred_class = []   # For confusion matrix (image-level)
    conf_threshold = 0.5  # Threshold for classification
    
    image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    for img_file in image_files[:200]:  # Use first 200 images (faster)
        img_path = os.path.join(val_images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(val_labels_dir, label_file)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # Load ground truth
        gt_boxes = load_yolo_labels(label_path, img_width, img_height)
        has_gt_person = len(gt_boxes) > 0
        
        # Get predictions
        preds = model(img, conf=0.001, classes=[0])[0]  # Low conf threshold for ROC
        pred_boxes = preds.boxes.xyxy.cpu().numpy() if preds.boxes is not None else []
        pred_confs = preds.boxes.conf.cpu().numpy() if preds.boxes is not None else []
        
        # Process for confusion matrix (image-level: has person vs not)
        has_pred_person = len(pred_boxes) > 0 and max(pred_confs) > conf_threshold if len(pred_confs) > 0 else False
        y_true_class.append(1 if has_gt_person else 0)
        y_pred_class.append(1 if has_pred_person else 0)
        
        # Process for ROC (detection-level)
        used_gt = set()
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= 0.5:
                y_true_binary.append(1)
                used_gt.add(best_gt_idx)
            else:
                y_true_binary.append(0)
            y_scores.append(pred_confs[i])
        
        # Add false negatives
        for j in range(len(gt_boxes)):
            if j not in used_gt:
                y_true_binary.append(1)
                y_scores.append(0.0)
    
    # Compute all metrics
    print("\n📊 Generating Plots and Metrics...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true_class, y_pred_class)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Person', 'Person'],
                yticklabels=['No Person', 'Person'],
                annot_kws={"size": 16})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    cm_plot_path = os.path.join(ML_DIR, "confusion_matrix_pretrained_yolov8n.png")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"📊 Confusion Matrix saved to: {cm_plot_path}")
    
    # 2. ROC Curve
    fpr = None
    tpr = None
    roc_auc = None
    if len(y_true_binary) > 0 and len(y_scores) > 0:
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#0066cc', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='#999999', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        roc_plot_path = os.path.join(ML_DIR, "roc_curve_pretrained_yolov8n.png")
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 ROC curve saved to: {roc_plot_path}")
    
    # 3. Compute Test Accuracy
    test_accuracy = (cm[0,0] + cm[1,1]) / cm.sum() if cm.sum() > 0 else 0
    
    # 4. Compute F1 from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision_cm = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_cm = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_cm = 2 * (precision_cm * recall_cm) / (precision_cm + recall_cm) if (precision_cm + recall_cm) > 0 else 0
    
    # Print all results nicely
    print("\n" + "="*65)
    print("📊 PRE-TRAINED YOLOv8n EVALUATION RESULTS")
    print("="*65)
    print(f"🎯 Test Accuracy:             {test_accuracy:.4f}")
    print(f"🎯 mAP50 (IoU=0.5):          {results.box.map50:.4f}")
    print(f"🎯 mAP50-95 (IoU=0.5-0.95):  {results.box.map:.4f}")
    print(f"✅ Precision (P):           {results.box.mp:.4f}")
    print(f"✅ Recall (R):              {results.box.mr:.4f}")
    f1_map = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
    print(f"✅ F1-Score (mAP-based):    {f1_map:.4f}")
    print(f"✅ F1-Score (CM-based):     {f1_cm:.4f}")
    if roc_auc is not None:
        print(f"📈 AUC:                     {roc_auc:.4f}")
    print("\n📊 Confusion Matrix Values:")
    print(f"   True Negative (TN):  {tn}")
    print(f"   False Positive (FP): {fp}")
    print(f"   False Negative (FN): {fn}")
    print(f"   True Positive (TP):  {tp}")
    print("="*65 + "\n")

    # Save all results to JSON
    metrics_file = os.path.join(ML_DIR, "pretrained_yolov8n_metrics.json")
    metrics_dict = {
        "test_accuracy": float(test_accuracy),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1_score_map_based": float(f1_map),
        "f1_score_cm_based": float(f1_cm),
        "confusion_matrix": cm.tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }
    if roc_auc is not None:
        metrics_dict["auc"] = float(roc_auc)
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"💾 All metrics saved to: {metrics_file}")
    print(f"📈 YOLO default plots saved to: runs/detect/val/")
    #print("\n✅ Done! All required metrics and graphs are ready for your report!")


if __name__ == "__main__":
    main()
