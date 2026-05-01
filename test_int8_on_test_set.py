"""
Test Int8 TFLite Model on 10% Held-out Test Set
"""
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import tensorflow as tf

# Paths
MODEL_DIR = "model_output/phase3_int8_optimized"
INT8_MODEL_PATH = os.path.join(MODEL_DIR, "wafer_classifier_int8.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
TEST_SET_PATH = os.path.join(MODEL_DIR, "test_set.json")

# Image settings
IMG_SIZE = 128

def load_and_preprocess_image_int8(image_path):
    """Load and preprocess image for int8 model"""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1] then scale for int8 quantization
    # Based on input quantization: scale=1.0, zero_point=-128
    # int8_value = float_value / scale + zero_point
    # For [0, 255] input normalized to [0, 1] and then to int8:
    img_normalized = img_array / 255.0
    
    # Convert to int8 range [-128, 127]
    # The model expects input where 0.0 maps to -128
    img_int8 = (img_normalized * 255 - 128).astype(np.int8)
    
    return img_int8.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def main():
    print("=" * 70)
    print("TESTING INT8 MODEL ON 10% HELD-OUT TEST SET")
    print("=" * 70)
    
    # Load labels
    with open(LABELS_PATH, 'r') as f:
        labels_list = json.load(f)
    idx_to_label = {i: label for i, label in enumerate(labels_list)}
    label_to_idx = {label: i for i, label in enumerate(labels_list)}
    print(f"\nClasses ({len(labels_list)}): {labels_list}")
    
    # Load test set
    with open(TEST_SET_PATH, 'r') as f:
        test_data = json.load(f)
    test_images = test_data['images']
    print(f"Test set size: {len(test_images)} images")
    
    # Load Int8 TFLite model
    print(f"\nLoading Int8 model: {INT8_MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=INT8_MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Get quantization parameters
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]
    
    print(f"\nInput quantization: scale={input_scale}, zero_point={input_zero_point}")
    print(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")
    
    # Test all images
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE")
    print("=" * 70)
    
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion_data = defaultdict(lambda: defaultdict(int))
    
    for i, img_path in enumerate(test_images):
        # Extract ground truth label from path
        parts = img_path.replace('\\', '/').split('/')
        true_label = parts[-2]  # Folder name is the label
        
        # Load and preprocess image
        full_path = img_path
        if not os.path.exists(full_path):
            print(f"Image not found: {full_path}")
            continue
        
        img_int8 = load_and_preprocess_image_int8(full_path)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_int8)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get prediction
        pred_idx = np.argmax(output[0])
        pred_label = idx_to_label[pred_idx]
        
        # Track metrics
        per_class_total[true_label] += 1
        confusion_data[true_label][pred_label] += 1
        
        if pred_label == true_label:
            correct += 1
            per_class_correct[true_label] += 1
        
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_images)} images...")
    
    # Results
    accuracy = 100.0 * correct / total if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 50)
    all_classes = sorted(set(list(per_class_total.keys()) + labels_list))
    
    for cls in all_classes:
        total_cls = per_class_total.get(cls, 0)
        correct_cls = per_class_correct.get(cls, 0)
        if total_cls > 0:
            acc = 100.0 * correct_cls / total_cls
            print(f"  {cls:15s}: {acc:6.2f}% ({correct_cls}/{total_cls})")
        else:
            print(f"  {cls:15s}: No samples")
    
    # Confusion matrix summary (top misclassifications)
    print("\n" + "=" * 70)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 70)
    
    misclassifications = []
    for true_cls, preds in confusion_data.items():
        for pred_cls, count in preds.items():
            if true_cls != pred_cls and count > 0:
                misclassifications.append((true_cls, pred_cls, count))
    
    misclassifications.sort(key=lambda x: x[2], reverse=True)
    
    if misclassifications:
        print("\nTop misclassifications (True -> Predicted: Count):")
        for true_cls, pred_cls, count in misclassifications[:15]:
            print(f"  {true_cls} -> {pred_cls}: {count}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {INT8_MODEL_PATH}")
    print(f"Test Set Size: {total} images")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Correct Predictions: {correct}/{total}")
    
    return accuracy

if __name__ == "__main__":
    main()
