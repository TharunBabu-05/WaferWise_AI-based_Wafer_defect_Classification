"""
Predict classes for Hackathon_phase3_prediction_dataset using Float32 TFLite model
"""
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import csv

# Paths
MODEL_DIR = "model_output/phase3_int8_optimized"
FLOAT32_MODEL_PATH = os.path.join(MODEL_DIR, "wafer_classifier_float32.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
PREDICTION_DATASET = "Hackathon_phase3_prediction_dataset"
OUTPUT_CSV = "phase3_predictions_float32.csv"

# Image settings
IMG_SIZE = 128

def load_and_preprocess_image(image_path):
    """Load and preprocess image for float32 model"""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    
    # Keep in [0, 255] range - MobileNetV3 handles internal normalization
    return img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def main():
    print("=" * 70)
    print("PREDICTING HACKATHON PHASE3 PREDICTION DATASET")
    print("=" * 70)
    
    # Load labels
    with open(LABELS_PATH, 'r') as f:
        labels_list = json.load(f)
    idx_to_label = {i: label for i, label in enumerate(labels_list)}
    print(f"\nClasses ({len(labels_list)}): {labels_list}")
    
    # Load Float32 TFLite model
    print(f"\nLoading Float32 model: {FLOAT32_MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=FLOAT32_MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Get all images
    image_files = sorted([f for f in os.listdir(PREDICTION_DATASET) if f.endswith('.png')],
                        key=lambda x: int(os.path.splitext(x)[0]))
    print(f"\nFound {len(image_files)} images to predict")
    
    # Run predictions
    print("\n" + "=" * 70)
    print("RUNNING PREDICTIONS")
    print("=" * 70)
    
    predictions = []
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(PREDICTION_DATASET, img_file)
        img_id = os.path.splitext(img_file)[0]  # Get filename without extension
        
        # Load and preprocess
        img_input = load_and_preprocess_image(img_path)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Get prediction
        pred_idx = np.argmax(output[0])
        pred_label = idx_to_label[pred_idx]
        confidence = float(output[0][pred_idx])
        
        predictions.append({
            'id': img_id,
            'filename': img_file,
            'predicted_class': pred_label,
            'confidence': confidence
        })
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    # Save predictions to CSV
    print(f"\n" + "=" * 70)
    print("SAVING PREDICTIONS")
    print("=" * 70)
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'predicted_class'])
        for pred in predictions:
            writer.writerow([pred['id'], pred['predicted_class']])
    
    print(f"Predictions saved to: {OUTPUT_CSV}")
    
    # Summary statistics
    class_counts = {}
    for pred in predictions:
        cls = pred['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"\nTotal images: {len(predictions)}")
    print(f"\nPrediction distribution:")
    print("-" * 40)
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = 100.0 * count / len(predictions)
        print(f"  {cls:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Show first 20 predictions
    print("\n" + "=" * 70)
    print("FIRST 20 PREDICTIONS")
    print("=" * 70)
    for pred in predictions[:20]:
        print(f"  {pred['id']:>5s} -> {pred['predicted_class']}")
    
    print(f"\nOutput file: {OUTPUT_CSV}")
    
    return predictions

if __name__ == "__main__":
    main()
