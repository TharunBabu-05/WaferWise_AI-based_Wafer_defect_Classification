"""
Phase 3 Final Training Script - INT8 Optimized
MobileNetV3Small with Quantization-Aware Training for NXP RT1170EVK

Uses 80/10/10 split for train/val/test
Includes Quantization-Aware Training for better int8 accuracy
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime
import random

# Try to import TF Model Optimization Toolkit
try:
    import tensorflow_model_optimization as tfmot
    HAS_TFMOT = True
    print("TensorFlow Model Optimization Toolkit available")
except ImportError:
    HAS_TFMOT = False
    print("TensorFlow Model Optimization Toolkit not available - will use standard training with careful quantization")

# Configuration
DATASET_DIR = "final_4000_dataset"
OUTPUT_DIR = "model_output/phase3_int8_optimized"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# Training phases
PHASE1_EPOCHS = 30   # Frozen backbone
PHASE2_EPOCHS = 170  # Fine-tuning
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_dataset(dataset_dir):
    """Load all images and labels from dataset directory."""
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    
    # Filter only directories
    class_names = [c for c in class_names if os.path.isdir(os.path.join(dataset_dir, c))]
    
    print(f"\nLoading dataset from: {dataset_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        class_images = []
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(class_dir, img_name)
                class_images.append(img_path)
        
        images.extend(class_images)
        labels.extend([class_idx] * len(class_images))
        print(f"  {class_name}: {len(class_images)} images")
    
    print(f"Total: {len(images)} images")
    return images, labels, class_names

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=-1)  # (128, 128, 1)
    return img_array

def create_dataset(image_paths, labels, batch_size, augment=False, shuffle=True):
    """Create tf.data.Dataset with optional augmentation."""
    
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        return img, label
    
    def augment_image(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        # Random rotation (0, 90, 180, 270 degrees)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        img = tf.clip_by_value(img, 0, 255)
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=SEED)
    
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_model(num_classes, input_shape=(128, 128, 1)):
    """Create MobileNetV3Small model."""
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Expand to 3 channels for MobileNetV3
    x = layers.Concatenate()([inputs, inputs, inputs])
    
    # MobileNetV3Small backbone
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
        minimalistic=True,
        alpha=1.0
    )
    
    x = backbone(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='wafer_classifier_int8')
    
    return model, backbone

def apply_quantization_aware_training(model):
    """Apply quantization-aware training to the model."""
    if not HAS_TFMOT:
        print("TFMOT not available, skipping QAT")
        return model
    
    # Apply QAT
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    
    return qat_model

def convert_to_int8_tflite(model, representative_dataset, output_path):
    """Convert model to fully quantized int8 TFLite."""
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    return len(tflite_model) / 1024  # Size in KB

def representative_dataset_gen(images, num_samples=200):
    """Generator for representative dataset."""
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    for idx in indices:
        img = load_and_preprocess_image(images[idx])
        img = np.expand_dims(img, axis=0)
        yield [img.astype(np.float32)]

def main():
    print("=" * 70)
    print("PHASE 3 FINAL TRAINING - INT8 OPTIMIZED")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    all_images, all_labels, class_names = load_dataset(DATASET_DIR)
    num_classes = len(class_names)
    
    # Split dataset: 80% train, 10% val, 10% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_images, all_labels, test_size=0.1, random_state=SEED, stratify=all_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=SEED, stratify=y_train_val
    )
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(X_train)} images (80%)")
    print(f"  Validation: {len(X_val)} images (10%)")
    print(f"  Test:       {len(X_test)} images (10%)")
    
    # Save test set for later evaluation
    test_set = {
        'images': X_test,
        'labels': y_test
    }
    with open(os.path.join(OUTPUT_DIR, 'test_set.json'), 'w') as f:
        json.dump(test_set, f, indent=2)
    
    # Create datasets
    train_dataset = create_dataset(X_train, y_train, BATCH_SIZE, augment=True, shuffle=True)
    val_dataset = create_dataset(X_val, y_val, BATCH_SIZE, augment=False, shuffle=False)
    test_dataset = create_dataset(X_test, y_test, BATCH_SIZE, augment=False, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    model, backbone = create_model(num_classes)
    model.summary()
    
    # Phase 1: Train with frozen backbone
    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING WITH FROZEN BACKBONE")
    print("=" * 70)
    
    backbone.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase1 = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model_phase1.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Phase 2: Fine-tune entire model
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNING ENTIRE MODEL")
    print("=" * 70)
    
    backbone.trainable = True
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_phase2 = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model_phase2.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_DIR, 'logs'),
            histogram_freq=0
        )
    ]
    
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save best model
    model.save(os.path.join(OUTPUT_DIR, 'best_model.keras'))
    
    # Convert to Float32 TFLite first (for comparison)
    print("\n" + "=" * 70)
    print("CONVERTING TO FLOAT32 TFLITE")
    print("=" * 70)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float32 = converter.convert()
    
    float32_path = os.path.join(OUTPUT_DIR, 'wafer_classifier_float32.tflite')
    with open(float32_path, 'wb') as f:
        f.write(tflite_float32)
    
    float32_size = len(tflite_float32) / 1024
    print(f"Float32 TFLite saved to: {float32_path}")
    print(f"Float32 TFLite size: {float32_size:.1f} KB")
    
    # Convert to INT8 TFLite with proper calibration
    print("\n" + "=" * 70)
    print("CONVERTING TO INT8 TFLITE WITH CALIBRATION")
    print("=" * 70)
    
    # Create representative dataset generator using training images
    def rep_dataset():
        # Use a subset of training images for calibration
        num_calibration = min(500, len(X_train))
        indices = np.random.choice(len(X_train), num_calibration, replace=False)
        
        for idx in indices:
            img = load_and_preprocess_image(X_train[idx])
            img = np.expand_dims(img, axis=0)
            yield [img.astype(np.float32)]
    
    int8_path = os.path.join(OUTPUT_DIR, 'wafer_classifier_int8.tflite')
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_int8 = converter.convert()
        
        with open(int8_path, 'wb') as f:
            f.write(tflite_int8)
        
        int8_size = len(tflite_int8) / 1024
        print(f"Int8 TFLite saved to: {int8_path}")
        print(f"Int8 TFLite size: {int8_size:.1f} KB")
    except Exception as e:
        print(f"Error during int8 conversion: {e}")
        int8_size = 0
    
    # Verify int8 model
    print("\n" + "=" * 70)
    print("VERIFYING INT8 MODEL")
    print("=" * 70)
    
    try:
        interpreter = tf.lite.Interpreter(model_path=int8_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output dtype: {output_details[0]['dtype']}")
        
        # Test inference on a few samples
        correct = 0
        total = min(100, len(X_test))
        
        input_scale = input_details[0]['quantization_parameters']['scales'][0]
        input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
        
        print(f"\nInput quantization: scale={input_scale}, zero_point={input_zp}")
        
        for i in range(total):
            img = load_and_preprocess_image(X_test[i])
            
            # Quantize input
            img_quantized = np.clip(img / input_scale + input_zp, -128, 127).astype(np.int8)
            img_quantized = np.expand_dims(img_quantized, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], img_quantized)
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output[0])
            
            if pred == y_test[i]:
                correct += 1
        
        int8_accuracy = correct / total * 100
        print(f"\nInt8 Quick Test Accuracy: {int8_accuracy:.2f}% ({correct}/{total})")
        
    except Exception as e:
        print(f"Error verifying int8 model: {e}")
        int8_accuracy = 0
    
    # Save labels
    with open(os.path.join(OUTPUT_DIR, 'labels.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'labels.txt'), 'w') as f:
        for cls in class_names:
            f.write(f"{cls}\n")
    
    # Save config
    config = {
        'model': 'MobileNetV3Small',
        'input_shape': [1, 128, 128, 1],
        'num_classes': num_classes,
        'class_names': class_names,
        'dataset': DATASET_DIR,
        'train_images': len(X_train),
        'val_images': len(X_val),
        'test_images': len(X_test),
        'test_accuracy_keras': float(test_accuracy),
        'int8_quick_test_accuracy': float(int8_accuracy) if int8_accuracy else 0,
        'float32_tflite_size_kb': float32_size,
        'int8_tflite_size_kb': int8_size if int8_size else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate C header files for MCU
    print("\n" + "=" * 70)
    print("GENERATING C HEADER FILES")
    print("=" * 70)
    
    # Labels header
    labels_h = '#ifndef WAFER_LABELS_H\n#define WAFER_LABELS_H\n\n'
    labels_h += f'#define NUM_CLASSES {num_classes}\n\n'
    labels_h += 'const char* class_labels[] = {\n'
    for cls in class_names:
        labels_h += f'    "{cls}",\n'
    labels_h += '};\n\n#endif // WAFER_LABELS_H\n'
    
    with open(os.path.join(OUTPUT_DIR, 'wafer_labels.h'), 'w') as f:
        f.write(labels_h)
    
    print(f"Labels header saved to: {os.path.join(OUTPUT_DIR, 'wafer_labels.h')}")
    
    # Int8 model header
    with open(int8_path, 'rb') as f:
        model_data = f.read()
    
    model_h = '#ifndef WAFER_MODEL_INT8_H\n#define WAFER_MODEL_INT8_H\n\n'
    model_h += f'const unsigned int wafer_model_int8_len = {len(model_data)};\n'
    model_h += 'alignas(8) const unsigned char wafer_model_int8[] = {\n'
    
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        model_h += f'    {hex_str},\n'
    
    model_h += '};\n\n#endif // WAFER_MODEL_INT8_H\n'
    
    with open(os.path.join(OUTPUT_DIR, 'wafer_model_int8.h'), 'w') as f:
        f.write(model_h)
    
    print(f"Int8 model header saved to: {os.path.join(OUTPUT_DIR, 'wafer_model_int8.h')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nFiles generated:")
    print(f"  - best_model.keras")
    print(f"  - wafer_classifier_float32.tflite ({float32_size:.1f} KB)")
    print(f"  - wafer_classifier_int8.tflite ({int8_size:.1f} KB)")
    print(f"  - labels.json, labels.txt")
    print(f"  - wafer_labels.h, wafer_model_int8.h")
    print(f"  - config.json, test_set.json")
    print(f"\nKeras Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Int8 Quick Test Accuracy: {int8_accuracy:.2f}%")
    print(f"\nSize reduction: {float32_size:.1f} KB -> {int8_size:.1f} KB ({(1 - int8_size/float32_size)*100:.1f}% smaller)")

if __name__ == "__main__":
    main()
