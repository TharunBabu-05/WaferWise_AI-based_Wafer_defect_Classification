"""
Test INT8 TFLite Model with INT8 Input Data
This simulates how the model will run on MCU with uint8 image input
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import time

# Paths
MODEL_PATH = r'c:\Users\tharu\IESA_FINAL_task_1\model_output\mcu_model_8\wafer_classifier_int8.tflite'
DATA_DIR = r'c:\Users\tharu\IESA_FINAL_task_1\hackathon_balanced_2000'
IMG_SIZE = 128

# Load class names
CLASS_NAMES = sorted(os.listdir(DATA_DIR))
print(f"Classes: {CLASS_NAMES}")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\nModel Info:")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Input quantization: {input_details[0].get('quantization', 'None')}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# Check if model accepts int8 input directly
input_dtype = input_details[0]['dtype']
print(f"\nModel expects {input_dtype} input")


def preprocess_image_uint8(image_path):
    """Load image and return as UINT8 [0-255]."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.uint8)  # Keep as uint8 [0-255]
    return img_array


def preprocess_for_model(img_uint8):
    """Convert uint8 image to model input format.
    Model has built-in Rescaling layer that handles [0,1] -> [-1,1]
    So we just need to normalize to [0,1]
    """
    # Input: uint8 [0-255] -> float32 [0,1]
    # The model's internal Rescaling layer will convert to [-1,1]
    img_float = img_uint8.astype(np.float32) / 255.0
    return np.expand_dims(img_float, axis=0)


def test_with_uint8_pipeline():
    """Test model simulating MCU uint8 input pipeline."""
    
    print("\n" + "="*70)
    print("Testing TFLite INT8 Model with UINT8 Input Pipeline")
    print("="*70)
    print("Simulating MCU: uint8 image -> preprocess -> model inference")
    
    y_true = []
    y_pred = []
    
    total_images = 0
    start_time = time.time()
    
    # Count total images
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(images)
    
    print(f"Total images: {total_images}")
    
    processed = 0
    class_stats = {name: {'correct': 0, 'total': 0} for name in CLASS_NAMES}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            
            # Load as uint8 (simulating MCU camera input)
            img_uint8 = preprocess_image_uint8(img_path)
            
            # Convert to model input format
            input_data = preprocess_for_model(img_uint8)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_idx = np.argmax(output_data[0])
            
            y_true.append(class_idx)
            y_pred.append(predicted_idx)
            
            class_stats[class_name]['total'] += 1
            if predicted_idx == class_idx:
                class_stats[class_name]['correct'] += 1
            
            processed += 1
            if processed % 200 == 0 or processed == total_images:
                print(f"\r  Processed: {processed}/{total_images} ({processed/total_images*100:.1f}%)", end="", flush=True)
    
    elapsed = time.time() - start_time
    print(f" - Done!")
    print(f"  Time: {elapsed:.2f}s ({total_images/elapsed:.1f} images/sec)")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS (UINT8 Input Pipeline)")
    print("="*70)
    
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for name in CLASS_NAMES:
        stats = class_stats[name]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        bar = '█' * int(acc * 20)
        print(f"  {name:10s}: {stats['correct']:3d}/{stats['total']:3d} = {acc*100:5.1f}% {bar}")
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    overall_acc = np.mean(y_true == y_pred)
    
    print("-" * 40)
    print(f"OVERALL ACCURACY: {overall_acc*100:.2f}% ({np.sum(y_true == y_pred)}/{len(y_true)})")
    print("-" * 40)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    header = "         " + " ".join([f"{name[:5]:>5s}" for name in CLASS_NAMES])
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name[:10]:10s}|" + " ".join([f"{cm[i,j]:5d}" for j in range(len(CLASS_NAMES))])
        print(row)
    
    return overall_acc


def test_quantization_comparison():
    """Compare float32 vs simulated int8 preprocessing."""
    
    print("\n" + "="*70)
    print("Quantization Precision Analysis")
    print("="*70)
    
    # Get a sample image
    sample_class = CLASS_NAMES[0]
    sample_dir = os.path.join(DATA_DIR, sample_class)
    sample_img = os.listdir(sample_dir)[0]
    sample_path = os.path.join(sample_dir, sample_img)
    
    # Load image
    img = Image.open(sample_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Float32 pipeline (original test)
    img_float32 = np.array(img, dtype=np.float32) / 255.0
    input_float32 = np.expand_dims(img_float32, axis=0)
    
    # Uint8 -> Float32 pipeline (MCU simulation)
    img_uint8 = np.array(img, dtype=np.uint8)
    img_converted = img_uint8.astype(np.float32) / 255.0
    img_scaled = (img_converted * 2.0) - 1.0
    input_mcu = np.expand_dims(img_scaled, axis=0)
    
    # Check differences
    print(f"\nSample image: {sample_path}")
    print(f"Original uint8 range: [{img_uint8.min()}, {img_uint8.max()}]")
    print(f"Float32 input range: [{input_float32.min():.4f}, {input_float32.max():.4f}]")
    print(f"MCU scaled input range: [{input_mcu.min():.4f}, {input_mcu.max():.4f}]")
    
    # Run both through model
    interpreter.set_tensor(input_details[0]['index'], input_float32)
    interpreter.invoke()
    output_float32 = interpreter.get_tensor(output_details[0]['index'])[0]
    
    interpreter.set_tensor(input_details[0]['index'], input_mcu)
    interpreter.invoke()
    output_mcu = interpreter.get_tensor(output_details[0]['index'])[0]
    
    print(f"\nPrediction comparison for sample image:")
    print(f"  Float32 [0,1] input -> Class: {CLASS_NAMES[np.argmax(output_float32)]} (conf: {output_float32.max():.4f})")
    print(f"  MCU [-1,1] input    -> Class: {CLASS_NAMES[np.argmax(output_mcu)]} (conf: {output_mcu.max():.4f})")
    
    # Note about the model's expected preprocessing
    print("\n" + "="*70)
    print("NOTE: mcu_model_8 uses MobileNetV2 which expects [-1, 1] input range")
    print("The model has built-in Rescaling layer that handles [0,1] -> [-1,1]")
    print("="*70)


if __name__ == '__main__':
    # Run comparison first
    test_quantization_comparison()
    
    # Test with uint8 pipeline
    accuracy = test_with_uint8_pipeline()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: mcu_model_8")
    print(f"Input: 128x128x3 (uint8 -> float32 preprocessed)")
    print(f"Accuracy with UINT8 input pipeline: {accuracy*100:.2f}%")
