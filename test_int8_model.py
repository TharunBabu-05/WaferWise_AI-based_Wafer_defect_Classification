"""
Test Int8 TFLite Model on 10% held-out test set
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict

# Configuration
MODEL_DIR = "model_output/phase3_final_float32"
TFLITE_PATH = os.path.join(MODEL_DIR, "wafer_classifier_int8.tflite")
TEST_SET_JSON = os.path.join(MODEL_DIR, "test_set.json")
LABELS_JSON = os.path.join(MODEL_DIR, "labels.json")
IMG_SIZE = (128, 128)

def load_and_preprocess_image_int8(image_path):
    """Load and preprocess a single image for int8 inference."""
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    img_array = np.array(img, dtype=np.int8)
    img_array = img_array.astype(np.int8)  # Ensure int8
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
    print("INT8 MODEL TEST - 10% HELD-OUT TEST SET")
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
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Get quantization parameters
    input_scale = input_details[0].get('quantization_parameters', {}).get('scales', [])
    input_zero_point = input_details[0].get('quantization_parameters', {}).get('zero_points', [])
    output_scale = output_details[0].get('quantization_parameters', {}).get('scales', [])
    output_zero_point = output_details[0].get('quantization_parameters', {}).get('zero_points', [])
    
    print(f"\nInput quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")
    
    model_size = os.path.getsize(TFLITE_PATH) / 1024
    print(f"Model size: {model_size:.1f} KB")
    
    # Determine input preprocessing based on input dtype
    input_dtype = input_details[0]['dtype']
    
    # Run inference on test set
    print("\n" + "-" * 70)
    print("RUNNING INFERENCE ON TEST SET")
    print("-" * 70)
    
    y_true = []
    y_pred = []
    y_confidence = []
    
    for img_path, true_label in zip(test_images, test_labels):
        try:
            # Load image
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            img = img.resize(IMG_SIZE, Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32)
            
            # Quantize input for int8: value_int8 = value_float / scale + zero_point
            # With scale=1 and zero_point=-128: int8_value = float_value - 128
            if input_dtype == np.int8:
                scale = input_scale[0] if len(input_scale) > 0 else 1.0
                zp = input_zero_point[0] if len(input_zero_point) > 0 else -128
                # For range 0-255 -> -128 to 127
                img_array = np.clip(img_array / scale + zp, -128, 127).astype(np.int8)
            elif input_dtype == np.uint8:
                img_array = img_array.astype(np.uint8)
            
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            
            output = run_inference(interpreter, img_array)
            
            pred_idx = np.argmax(output[0])
            confidence = output[0][pred_idx]
            
            y_true.append(true_label)
            y_pred.append(pred_idx)
            y_confidence.append(float(confidence))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    
    per_class_metrics = []
    for i, cls in enumerate(class_names):
        mask = y_true == i
        cls_total = np.sum(mask)
        cls_correct = np.sum((y_true == i) & (y_pred == i))
        cls_acc = cls_correct / cls_total * 100 if cls_total > 0 else 0
        print(f"  {cls:12s}: {cls_acc:5.1f}% ({cls_correct:3d}/{cls_total:3d})")
        per_class_metrics.append({
            'class': cls,
            'accuracy': cls_acc,
            'correct': int(cls_correct),
            'total': int(cls_total)
        })
    
    # Precision, Recall, F1
    print("\nPer-Class Precision, Recall, F1:")
    print("-" * 70)
    print(f"{'Class':13s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    
    f1_scores = []
    supports = []
    
    for i, cls in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = int(np.sum(y_true == i))
        
        f1_scores.append(f1)
        supports.append(support)
        
        print(f"  {cls:12s} {precision:9.1f}% {recall:9.1f}% {f1:9.1f}% {support:10d}")
    
    weighted_f1 = np.average(f1_scores, weights=supports) if sum(supports) > 0 else 0
    print("-" * 70)
    print(f"  {'Weighted Avg':12s} {'-':>10s} {accuracy:9.1f}% {weighted_f1:9.1f}% {total:10d}")
    
    # Confusion matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    
    # Print header
    header = "True\\Pred  " + " ".join([f"{cls[:5]:>5s}" for cls in class_names])
    print(header)
    print("-" * len(header))
    
    for i, cls in enumerate(class_names):
        row = f"{cls[:10]:10s}|"
        for j in range(n_classes):
            if confusion[i, j] == 0:
                row += "    ."
            else:
                row += f"{confusion[i, j]:5d}"
        print(row)
    
    # Save results
    results = {
        'model': 'wafer_classifier_int8.tflite',
        'model_size_kb': model_size,
        'test_images': total,
        'overall_accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': confusion.tolist(),
        'class_names': class_names
    }
    
    results_file = os.path.join(MODEL_DIR, "int8_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
