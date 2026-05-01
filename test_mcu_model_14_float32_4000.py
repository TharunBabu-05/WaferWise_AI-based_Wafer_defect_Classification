"""
Test script for mcu_model_14_float32_4000 TFLite model
Tests on hackathon_test_dataset
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict

# Configuration
MODEL_PATH = "model_output/mcu_model_14_float32_4000/wafer_classifier_float32.tflite"
TEST_DATASET = "hackathon_test_dataset"
IMG_SIZE = (128, 128)

# Class labels (alphabetical order as used during training)
CLASS_NAMES = ['Bridge', 'CMP', 'Clean', 'Crack', 'LER', 'Open', 'Other', 'Particle', 'VIA']

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for inference."""
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to model input size
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    
    # Convert to numpy array and add batch/channel dimensions
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (128, 128, 1)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension (1, 128, 128, 1)
    
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
    print("=" * 60)
    print("MCU_MODEL_14_FLOAT32_4000 TEST")
    print("=" * 60)
    
    # Load TFLite model
    print(f"\nLoading model: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Get model info
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Get model file size
    model_size = os.path.getsize(MODEL_PATH) / 1024
    print(f"Model size: {model_size:.1f} KB")
    
    print(f"\nTest dataset: {TEST_DATASET}")
    print("-" * 60)
    
    # Track results
    total_correct = 0
    total_images = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    misclassified = []
    
    # Process each class folder
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_DATASET, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
        
        # Get all images in the class folder
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Load and preprocess
                input_data = load_and_preprocess_image(img_path)
                
                # Run inference
                output = run_inference(interpreter, input_data)
                predicted_idx = np.argmax(output[0])
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence = output[0][predicted_idx]
                
                # Update statistics
                total_images += 1
                class_total[class_name] += 1
                confusion_matrix[class_name][predicted_class] += 1
                
                if predicted_class == class_name:
                    total_correct += 1
                    class_correct[class_name] += 1
                else:
                    misclassified.append({
                        'file': img_path,
                        'true': class_name,
                        'predicted': predicted_class,
                        'confidence': confidence
                    })
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({total_correct}/{total_images})")
    
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for class_name in CLASS_NAMES:
        correct = class_correct[class_name]
        total = class_total[class_name]
        if total > 0:
            class_acc = correct / total * 100
            print(f"  {class_name:12s}: {class_acc:5.1f}% ({correct:3d}/{total:3d})")
        else:
            print(f"  {class_name:12s}: N/A (no samples)")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    header = "True\\Pred  " + " ".join([f"{c[:4]:>5s}" for c in CLASS_NAMES])
    print(header)
    print("-" * len(header))
    
    for true_class in CLASS_NAMES:
        row = f"{true_class[:9]:9s} |"
        for pred_class in CLASS_NAMES:
            count = confusion_matrix[true_class][pred_class]
            if count > 0:
                row += f" {count:4d}"
            else:
                row += "    ."
        print(row)
    
    # Print some misclassified examples
    if misclassified:
        print(f"\nMisclassified Examples (showing first 10):")
        print("-" * 60)
        for i, item in enumerate(misclassified[:10]):
            print(f"  {os.path.basename(item['file'])}")
            print(f"    True: {item['true']}, Predicted: {item['predicted']} ({item['confidence']:.2%})")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return accuracy

if __name__ == "__main__":
    main()
