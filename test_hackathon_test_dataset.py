"""
Test Phase 3 Final Float32 Model on hackathon_test_dataset
Handles mapping between 9-class test set and 11-class model
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict

# Configuration
MODEL_DIR = "model_output/phase3_final_float32"
TFLITE_PATH = os.path.join(MODEL_DIR, "wafer_classifier_float32.tflite")
LABELS_JSON = os.path.join(MODEL_DIR, "labels.json")
TEST_DATASET = "Hackathon_phase3_pridicton_dataset"
IMG_SIZE = (128, 128)

# Model classes (11 classes)
MODEL_CLASSES = [
    "BRIDGE", "CLEAN_CRACK", "CLEAN_LAYER", "CLEAN_VIA", "CMP",
    "CRACK", "LER", "OPEN", "OTHERS", "PARTICLE", "VIA"
]

# Test dataset classes (9 classes) - mapped to model indices
# Note: "Clean" in test data maps to CLEAN_CRACK, CLEAN_LAYER, CLEAN_VIA (indices 1, 2, 3)
TEST_CLASS_MAPPING = {
    "Bridge": [0],      # BRIDGE
    "Clean": [1, 2, 3], # CLEAN_CRACK, CLEAN_LAYER, CLEAN_VIA (any is correct)
    "CMP": [4],         # CMP
    "Crack": [5],       # CRACK
    "LER": [6],         # LER
    "Open": [7],        # OPEN
    "Other": [8],       # OTHERS
    "Particle": [9],    # PARTICLE
    "VIA": [10],        # VIA
}

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for inference."""
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=-1)  # (128, 128, 1)
    img_array = np.expand_dims(img_array, axis=0)   # (1, 128, 128, 1)
    return img_array

def run_inference(interpreter, input_data):
    """Run inference on the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def load_test_dataset():
    """Load test images from hackathon_test_dataset folder."""
    test_images = []
    test_labels = []
    test_class_names = []
    
    for class_name in sorted(os.listdir(TEST_DATASET)):
        class_dir = os.path.join(TEST_DATASET, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(class_dir, img_name)
                test_images.append(img_path)
                test_labels.append(class_name)
                test_class_names.append(class_name)
    
    return test_images, test_labels

def main():
    print("=" * 70)
    print("PHASE 3 FINAL MODEL TEST - HACKATHON TEST DATASET")
    print("=" * 70)
    
    # Load model classes
    with open(LABELS_JSON, 'r') as f:
        model_classes = json.load(f)
    print(f"\nModel Classes ({len(model_classes)}): {model_classes}")
    
    # Load test dataset
    test_images, test_labels = load_test_dataset()
    print(f"\nTest Dataset: {TEST_DATASET}")
    print(f"Total test images: {len(test_images)}")
    
    # Count per class
    class_counts = defaultdict(int)
    for label in test_labels:
        class_counts[label] += 1
    
    print("\nPer-class distribution:")
    for cls in sorted(class_counts.keys()):
        print(f"  {cls}: {class_counts[cls]} images")
    
    # Load TFLite model
    print(f"\nLoading model: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    
    # Get model info
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    model_size = os.path.getsize(TFLITE_PATH) / 1024
    print(f"Model size: {model_size:.1f} KB")
    
    # Run inference
    print("\n" + "-" * 70)
    print("RUNNING INFERENCE ON HACKATHON TEST DATASET")
    print("-" * 70)
    
    results = []
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    # For confusion matrix - track what Clean images were predicted as
    clean_predictions = defaultdict(int)
    
    # Track all predictions for detailed analysis
    all_predictions = []
    
    for img_path, true_class in zip(test_images, test_labels):
        try:
            input_data = load_and_preprocess_image(img_path)
            output = run_inference(interpreter, input_data)
            
            pred_idx = int(np.argmax(output[0]))
            confidence = float(output[0][pred_idx])
            pred_class = model_classes[pred_idx]
            
            # Check if prediction is correct based on mapping
            valid_indices = TEST_CLASS_MAPPING.get(true_class, [])
            is_correct = pred_idx in valid_indices
            
            per_class_total[true_class] += 1
            if is_correct:
                per_class_correct[true_class] += 1
            
            # Track Clean predictions
            if true_class == "Clean":
                clean_predictions[pred_class] += 1
            
            all_predictions.append({
                'image': os.path.basename(img_path),
                'true_class': true_class,
                'pred_class': pred_class,
                'pred_idx': pred_idx,
                'confidence': confidence,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate metrics
    total_correct = sum(per_class_correct.values())
    total_images = sum(per_class_total.values())
    overall_accuracy = total_correct / total_images * 100
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_images})")
    
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    for cls in sorted(per_class_total.keys()):
        correct = per_class_correct[cls]
        total = per_class_total[cls]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {cls:12s}: {acc:5.1f}% ({correct:3d}/{total:3d})")
    
    # Show Clean class prediction breakdown
    print("\n" + "-" * 70)
    print("CLEAN CLASS PREDICTION BREAKDOWN")
    print("-" * 70)
    print("(Clean images in test set were predicted as:)")
    clean_total = per_class_total.get("Clean", 0)
    for pred_class, count in sorted(clean_predictions.items(), key=lambda x: -x[1]):
        pct = count / clean_total * 100 if clean_total > 0 else 0
        correct_marker = "✓" if pred_class in ["CLEAN_CRACK", "CLEAN_LAYER", "CLEAN_VIA"] else "✗"
        print(f"  {pred_class:15s}: {count:3d} ({pct:5.1f}%) {correct_marker}")
    
    # Calculate precision, recall, F1 for each test class
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS (Test Dataset Classes)")
    print("-" * 70)
    print(f"{'Class':12s} {'Accuracy':>10s} {'Support':>10s}")
    
    metrics_data = []
    for cls in sorted(per_class_total.keys()):
        correct = per_class_correct[cls]
        total = per_class_total[cls]
        acc = correct / total * 100 if total > 0 else 0
        print(f"{cls:12s} {acc:9.1f}% {total:10d}")
        metrics_data.append({
            'class': cls,
            'accuracy': acc,
            'correct': correct,
            'total': total
        })
    
    # Show misclassified samples
    print("\n" + "-" * 70)
    print("MISCLASSIFIED SAMPLES")
    print("-" * 70)
    
    misclassified = [p for p in all_predictions if not p['correct']]
    print(f"Total misclassified: {len(misclassified)}/{total_images}")
    
    if misclassified:
        print("\nSample misclassifications (up to 20):")
        for p in misclassified[:20]:
            print(f"  {p['image']:40s} True: {p['true_class']:10s} -> Pred: {p['pred_class']:15s} (conf: {p['confidence']:.2f})")
    
    # Build confusion matrix for 9-class mapping
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (9x9 - Test Dataset Classes)")
    print("=" * 70)
    
    test_classes = sorted(per_class_total.keys())
    n_classes = len(test_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    # Map predictions to 9-class simplified form
    def map_pred_to_test_class(pred_class):
        if pred_class in ["CLEAN_CRACK", "CLEAN_LAYER", "CLEAN_VIA"]:
            return "Clean"
        elif pred_class == "BRIDGE":
            return "Bridge"
        elif pred_class == "CMP":
            return "CMP"
        elif pred_class == "CRACK":
            return "Crack"
        elif pred_class == "LER":
            return "LER"
        elif pred_class == "OPEN":
            return "Open"
        elif pred_class == "OTHERS":
            return "Other"
        elif pred_class == "PARTICLE":
            return "Particle"
        elif pred_class == "VIA":
            return "VIA"
        return None
    
    test_class_to_idx = {cls: i for i, cls in enumerate(test_classes)}
    
    for p in all_predictions:
        true_idx = test_class_to_idx.get(p['true_class'])
        pred_mapped = map_pred_to_test_class(p['pred_class'])
        pred_idx = test_class_to_idx.get(pred_mapped)
        
        if true_idx is not None and pred_idx is not None:
            confusion[true_idx, pred_idx] += 1
    
    # Print confusion matrix
    header = "True\\Pred  " + " ".join([f"{cls[:7]:>7s}" for cls in test_classes])
    print(header)
    print("-" * len(header))
    
    for i, cls in enumerate(test_classes):
        row_str = f"{cls[:10]:10s}|"
        for j in range(n_classes):
            if confusion[i, j] == 0:
                row_str += "      ."
            else:
                row_str += f"{confusion[i, j]:7d}"
        print(row_str)
    
    # Calculate precision and recall from confusion matrix
    print("\n" + "-" * 70)
    print("PRECISION & RECALL (9-Class)")
    print("-" * 70)
    print(f"{'Class':12s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s} {'Support':>10s}")
    
    f1_scores = []
    supports = []
    for i, cls in enumerate(test_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp  # column sum minus diagonal
        fn = confusion[i, :].sum() - tp  # row sum minus diagonal
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = confusion[i, :].sum()
        
        f1_scores.append(f1)
        supports.append(support)
        
        print(f"{cls:12s} {precision:11.1f}% {recall:11.1f}% {f1:11.1f}% {support:10d}")
    
    # Weighted average
    weighted_f1 = np.average(f1_scores, weights=supports) if sum(supports) > 0 else 0
    print(f"\n{'Weighted Avg':12s} {'-':>12s} {'-':>12s} {weighted_f1:11.1f}% {sum(supports):10d}")
    
    # Save results
    results_data = {
        'test_dataset': TEST_DATASET,
        'model_path': TFLITE_PATH,
        'model_size_kb': model_size,
        'total_images': total_images,
        'overall_accuracy': overall_accuracy,
        'per_class_metrics': metrics_data,
        'clean_predictions': dict(clean_predictions),
        'misclassified_count': len(misclassified),
        'confusion_matrix': confusion.tolist(),
        'class_order': test_classes,
        'weighted_f1': weighted_f1
    }
    
    results_file = os.path.join(MODEL_DIR, "hackathon_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
