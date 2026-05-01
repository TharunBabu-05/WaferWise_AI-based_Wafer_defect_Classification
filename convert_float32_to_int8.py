"""
Convert mcu_model_13_float32 from Float32 to INT8 quantization
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Configuration
MODEL_DIR = "model_output/phase3_final_float32"
KERAS_MODEL = os.path.join(MODEL_DIR, "best_model.keras")
OUTPUT_TFLITE = os.path.join(MODEL_DIR, "wafer_classifier_int8.tflite")
CALIBRATION_DATASET = "final_4000_dataset"  # Use training data for calibration
IMG_SIZE = (128, 128)
NUM_CALIBRATION_SAMPLES = 300  # Number of images for calibration

def load_calibration_images():
    """Load representative images for INT8 calibration."""
    images = []
    
    # Get class folders
    class_dirs = [d for d in os.listdir(CALIBRATION_DATASET) 
                  if os.path.isdir(os.path.join(CALIBRATION_DATASET, d))]
    
    samples_per_class = NUM_CALIBRATION_SAMPLES // len(class_dirs)
    
    for class_name in class_dirs:
        class_dir = os.path.join(CALIBRATION_DATASET, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Take samples from this class
        for img_file in image_files[:samples_per_class]:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path)
                if img.mode != 'L':
                    img = img.convert('L')
                img = img.resize(IMG_SIZE, Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=-1)  # (128, 128, 1)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} calibration images")
    return images

def representative_dataset_gen():
    """Generator for representative dataset used in quantization."""
    calibration_images = load_calibration_images()
    for img in calibration_images:
        # Add batch dimension
        yield [np.expand_dims(img, axis=0)]

def main():
    print("=" * 60)
    print("FLOAT32 TO INT8 CONVERSION")
    print("=" * 60)
    
    # Load the Keras model
    print(f"\nLoading Keras model: {KERAS_MODEL}")
    model = tf.keras.models.load_model(KERAS_MODEL)
    model.summary()
    
    # Create TFLite converter
    print("\nConverting to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags for full INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Force INT8 for all ops (full integer quantization)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert
    print("Running quantization with calibration data...")
    tflite_model = converter.convert()
    
    # Save INT8 model
    with open(OUTPUT_TFLITE, 'wb') as f:
        f.write(tflite_model)
    
    int8_size = os.path.getsize(OUTPUT_TFLITE) / 1024
    print(f"\nINT8 model saved: {OUTPUT_TFLITE}")
    print(f"INT8 model size: {int8_size:.1f} KB")
    
    # Compare with original float32
    float32_path = os.path.join(MODEL_DIR, "wafer_classifier_float32.tflite")
    if os.path.exists(float32_path):
        float32_size = os.path.getsize(float32_path) / 1024
        print(f"\nOriginal Float32 size: {float32_size:.1f} KB")
        print(f"Size reduction: {(1 - int8_size/float32_size)*100:.1f}%")
    
    # Verify the INT8 model
    print("\nVerifying INT8 model...")
    interpreter = tf.lite.Interpreter(model_path=OUTPUT_TFLITE)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Input scale: {input_details[0]['quantization'][0]}")
    print(f"Input zero_point: {input_details[0]['quantization'][1]}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output dtype: {output_details[0]['dtype']}")
    
    # Generate C header for INT8 model
    print("\nGenerating C header...")
    header_path = os.path.join(MODEL_DIR, "wafer_model_int8.h")
    
    with open(header_path, 'w') as f:
        f.write("// Auto-generated INT8 TFLite model header\n")
        f.write("// Input: 128x128x1 INT8 (grayscale)\n")
        f.write("// Output: 9 classes INT8\n\n")
        f.write("#ifndef WAFER_MODEL_INT8_H\n")
        f.write("#define WAFER_MODEL_INT8_H\n\n")
        f.write(f"const unsigned int wafer_model_int8_len = {len(tflite_model)};\n")
        f.write("const unsigned char wafer_model_int8[] = {\n")
        
        # Write model bytes
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        
        f.write("\n};\n\n")
        f.write("#endif  // WAFER_MODEL_INT8_H\n")
    
    header_size = os.path.getsize(header_path) / 1024
    print(f"C header saved: {header_path} ({header_size:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
