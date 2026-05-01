"""
Test script for Phase 3 Final Float32 Model
Tests on the 10% held-out test set
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
TEST_SET_JSON = os.path.join(MODEL_DIR, "test_set.json")
LABELS_JSON = os.path.join(MODEL_DIR, "labels.json")
IMG_SIZE = (128, 128)

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

def main():
    print("=" * 70)
    print("PHASE 3 FINAL MODEL TEST - 10% HELD-OUT TEST SET")
    print("=" * 70)
    
    # Load class names
    with open(LABELS_JSON, 'r') as f:
        class_names = json.load(f)
    print(f"\nClasses ({len(class_names)}): {class_names}")
    
    # Load test set
    with open(TEST_SET_JSON, 'r') as f:
        test_data = json.load(f)
    
    test_images = test_data['images']
    test_labels = test_data['labels']
    print(f"Test set: {len(test_images)} images")
    
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
    
    # Run inference on test set
    print("\n" + "-" * 70)
    print("RUNNING INFERENCE ON TEST SET")
    print("-" * 70)
    
    y_true = []
    y_pred = []
    y_confidence = []
    
    for img_path, true_label in zip(test_images, test_labels):
        try:
            input_data = load_and_preprocess_image(img_path)
            output = run_inference(interpreter, input_data)
            
            pred_idx = np.argmax(output[0])
            confidence = output[0][pred_idx]
            
            y_true.append(true_label)
            y_pred.append(pred_idx)
            y_confidence.append(confidence)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Per-class metrics
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    
    class_tp = defaultdict(int)  # True Positives
    class_fp = defaultdict(int)  # False Positives
    class_fn = defaultdict(int)  # False Negatives
    class_total = defaultdict(int)
    
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_tp[true] += 1
        else:
            class_fn[true] += 1
            class_fp[pred] += 1
    
    for i, class_name in enumerate(class_names):
        tp = class_tp[i]
        total_class = class_total[i]
        if total_class > 0:
            acc = tp / total_class * 100
            print(f"  {class_name:12s}: {acc:5.1f}% ({tp:3d}/{total_class:3d})")
        else:
            print(f"  {class_name:12s}: N/A (no samples)")
    
    # Precision, Recall, F1 per class
    print("\nPer-Class Precision, Recall, F1:")
    print("-" * 70)
    print(f"{'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for i, class_name in enumerate(class_names):
        tp = class_tp[i]
        fp = class_fp[i]
        fn = class_fn[i]
        support = class_total[i]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        
        print(f"  {class_name:12s} {precision*100:9.1f}% {recall*100:9.1f}% {f1*100:9.1f}% {support:10d}")
    
    # Weighted average
    total_support = sum(supports)
    weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
    weighted_recall = sum(r * s for r, s in zip(recalls, supports)) / total_support
    weighted_f1 = sum(f * s for f, s in zip(f1_scores, supports)) / total_support
    
    print("-" * 70)
    print(f"  {'Weighted Avg':12s} {weighted_precision*100:9.1f}% {weighted_recall*100:9.1f}% {weighted_f1*100:9.1f}% {total_support:10d}")
    
    # Confusion Matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    
    conf_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        conf_matrix[true][pred] += 1
    
    # Print header
    short_names = [c[:5] for c in class_names]
    header = "True\\Pred   " + " ".join([f"{n:>6s}" for n in short_names])
    print(header)
    print("-" * len(header))
    
    for i, class_name in enumerate(class_names):
        row = f"{class_name[:10]:10s} |"
        for j in range(len(class_names)):
            count = conf_matrix[i][j]
            if count > 0:
                row += f" {count:5d}"
            else:
                row += "     ."
        print(row)
    
    # Save detailed results
    results = {
        'accuracy': accuracy,
        'total_images': total,
        'correct_predictions': int(correct),
        'weighted_precision': weighted_precision * 100,
        'weighted_recall': weighted_recall * 100,
        'weighted_f1': weighted_f1 * 100,
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        results['per_class'][class_name] = {
            'precision': precisions[i] * 100,
            'recall': recalls[i] * 100,
            'f1': f1_scores[i] * 100,
            'support': supports[i]
        }
    
    results_path = os.path.join(MODEL_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
