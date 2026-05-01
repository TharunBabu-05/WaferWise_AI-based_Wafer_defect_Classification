"""
Test Pure INT8 Grayscale TFLite Model on Full Dataset
Model: mcu_model_12
Input: 128x128x1 INT8 grayscale
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

# Configuration
MODEL_PATH = Path("model_output/mcu_model_12_int8_pakka/wafer_classifier_int8.tflite")
DATA_DIR = Path("hackathon_test_dataset")
IMG_SIZE = 128

print("=" * 60)
print("Testing Pure INT8 Grayscale Model on Full Dataset")
print("=" * 60)

# Load class names
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(f"Classes: {class_names}")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"\nModel: {MODEL_PATH}")
print(f"Input shape: {input_details['shape']}")
print(f"Input dtype: {input_details['dtype']}")
print(f"Output shape: {output_details['shape']}")
print(f"Output dtype: {output_details['dtype']}")

# Get quantization parameters
input_scale = input_details['quantization_parameters']['scales'][0]
input_zero_point = input_details['quantization_parameters']['zero_points'][0]
output_scale = output_details['quantization_parameters']['scales'][0]
output_zero_point = output_details['quantization_parameters']['zero_points'][0]

print(f"\nInput quantization: scale={input_scale}, zero_point={input_zero_point}")
print(f"Output quantization: scale={output_scale}, zero_point={output_zero_point}")

# Load full dataset as RGB then convert to grayscale
full_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    shuffle=False,
    color_mode="rgb",
    label_mode="int"
)

# Convert to grayscale
def rgb_to_grayscale(image, label):
    gray = tf.image.rgb_to_grayscale(image)
    return gray, label

full_ds_gray = full_ds.map(rgb_to_grayscale)

print(f"\nTotal images: {len(full_ds)}")
print("\n" + "=" * 60)
print("TESTING...")
print("=" * 60)

correct = 0
total = 0
class_correct = {name: 0 for name in class_names}
class_total = {name: 0 for name in class_names}

# Confusion matrix
confusion = np.zeros((len(class_names), len(class_names)), dtype=int)

for images, labels in full_ds_gray:
    # Image is [1, 128, 128, 1] float32 [0, 255]
    img_float = images.numpy()[0]
    
    # Quantize to INT8
    img_int8 = np.clip(
        np.round(img_float / input_scale + input_zero_point), 
        -128, 127
    ).astype(np.int8)
    img_int8 = np.expand_dims(img_int8, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details['index'], img_int8)
    interpreter.invoke()
    
    # Get INT8 output and dequantize
    output_int8 = interpreter.get_tensor(output_details['index'])[0]
    output_float = (output_int8.astype(np.float32) - output_zero_point) * output_scale
    
    pred = np.argmax(output_float)
    label = labels.numpy()[0]
    
    class_name = class_names[label]
    class_total[class_name] += 1
    confusion[label, pred] += 1
    
    if pred == label:
        correct += 1
        class_correct[class_name] += 1
    total += 1
    
    if total % 200 == 0:
        print(f"  Processed {total}/{len(full_ds)} images...")

# Results
accuracy = correct / total * 100

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

print("\nPer-Class Accuracy:")
print("-" * 40)
for name in class_names:
    if class_total[name] > 0:
        cls_acc = class_correct[name] / class_total[name] * 100
        print(f"  {name:12s}: {cls_acc:5.1f}% ({class_correct[name]:3d}/{class_total[name]:3d})")

# Print confusion matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print("\nPredicted ->")
print("Actual |", end="")
for name in class_names:
    print(f" {name[:5]:>5}", end="")
print()
print("-" * (10 + 6 * len(class_names)))

for i, name in enumerate(class_names):
    print(f"{name[:8]:>8} |", end="")
    for j in range(len(class_names)):
        print(f" {confusion[i,j]:5d}", end="")
    print()

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model: mcu_model_12 (MobileNetV3Small)")
print(f"Input: 128x128x1 INT8 grayscale")
print(f"Output: 9 classes INT8")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{total}")

if accuracy >= 80:
    print("\n*** TARGET ACHIEVED: >80% ***")
