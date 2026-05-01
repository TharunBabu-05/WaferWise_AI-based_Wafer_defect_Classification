"""
MobileNetV3Small Float32 Training for Phase 3 Finals
Dataset: final_4000_dataset (11 classes)
Split: 80% train, 10% validation, 10% test
Output: Float32 TFLite
"""

import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import shutil

# Configuration
DATASET_DIR = "final_4000_dataset"
OUTPUT_DIR = "model_output/phase3_final_float32"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
PHASE1_EPOCHS = 30  # Frozen backbone
PHASE2_EPOCHS = 170  # Fine-tuning (total 200 epochs)
EARLY_STOPPING_PATIENCE = 30
REDUCE_LR_PATIENCE = 10

# Splits
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

def create_model(num_classes):
    """Create MobileNetV3Small model for grayscale input."""
    
    # Input layer for grayscale images [0, 255]
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), name='input')
    
    # Normalize to [-1, 1] for MobileNetV3
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1.0, name='normalize')(inputs)
    
    # Convert grayscale to RGB by concatenating 3 times
    x = tf.keras.layers.Concatenate(name='gray_to_rgb')([x, x, x])
    
    # Load MobileNetV3Small backbone
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet',
        minimalistic=True,
        alpha=1.0
    )
    backbone.trainable = False  # Freeze initially
    
    x = backbone(x)
    
    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense1')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout2')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = tf.keras.Model(inputs, outputs, name='MobileNetV3Small_Phase3_Float32')
    return model, backbone

def load_and_split_dataset():
    """Load dataset and split into train/val/test."""
    
    # Get all class names
    class_names = sorted([d for d in os.listdir(DATASET_DIR) 
                          if os.path.isdir(os.path.join(DATASET_DIR, d))])
    num_classes = len(class_names)
    
    print(f"Found {num_classes} classes: {class_names}")
    
    # Collect all image paths and labels
    all_images = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATASET_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        for img_file in image_files:
            all_images.append(os.path.join(class_dir, img_file))
            all_labels.append(class_idx)
    
    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split
    total = len(all_images)
    train_end = int(total * TRAIN_SPLIT)
    val_end = int(total * (TRAIN_SPLIT + VAL_SPLIT))
    
    train_images = all_images[:train_end]
    train_labels = all_labels[:train_end]
    
    val_images = all_images[train_end:val_end]
    val_labels = all_labels[train_end:val_end]
    
    test_images = all_images[val_end:]
    test_labels = all_labels[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images ({TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation: {len(val_images)} images ({VAL_SPLIT*100:.0f}%)")
    print(f"  Test: {len(test_images)} images ({TEST_SPLIT*100:.0f}%)")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), class_names

def load_image(path, label):
    """Load and preprocess a single image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)  # Keep in [0, 255] range
    return img, label

def augment_image(image, label):
    """Apply data augmentation."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    # Random rotation (approximate with crop and resize)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image, k=tf.random.uniform((), 0, 4, dtype=tf.int32))
    image = tf.clip_by_value(image, 0, 255)
    return image, label

def create_dataset(images, labels, batch_size, augment=False, shuffle=True):
    """Create tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images))
    
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def compute_class_weights(labels, num_classes):
    """Compute balanced class weights."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = {}
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total / (num_classes * counts[i])
        else:
            weights[i] = 1.0
    return weights

def main():
    print("=" * 60)
    print("PHASE 3 FINALS - MOBILENETV3 FLOAT32 TRAINING")
    print("=" * 60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Set random seed
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and split dataset
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels), class_names = load_and_split_dataset()
    num_classes = len(class_names)
    
    # Save class names
    with open(os.path.join(OUTPUT_DIR, 'labels.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, 'labels.txt'), 'w') as f:
        f.write('\n'.join(class_names))
    
    # Save test set info for later evaluation
    test_info = {'images': test_images, 'labels': test_labels}
    with open(os.path.join(OUTPUT_DIR, 'test_set.json'), 'w') as f:
        json.dump(test_info, f, indent=2)
    
    # Create datasets
    train_ds = create_dataset(train_images, train_labels, BATCH_SIZE, augment=True, shuffle=True)
    val_ds = create_dataset(val_images, val_labels, BATCH_SIZE, augment=False, shuffle=False)
    test_ds = create_dataset(test_images, test_labels, BATCH_SIZE, augment=False, shuffle=False)
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels, num_classes)
    print(f"\nClass weights: {class_weights}")
    
    # Create model
    print("\nCreating MobileNetV3Small model...")
    model, backbone = create_model(num_classes)
    model.summary()
    
    # Callbacks
    checkpoint_path = os.path.join(OUTPUT_DIR, 'best_model.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen backbone
    print("\n" + "=" * 60)
    print("PHASE 1: Training with frozen backbone")
    print("=" * 60)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune entire model
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning entire model")
    print("=" * 60)
    
    backbone.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        initial_epoch=len(history1.history['loss']),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    print("\nLoading best model...")
    model = tf.keras.models.load_model(checkpoint_path)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions for confusion matrix
    y_pred = []
    y_true = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean() * 100
            print(f"  {class_name}: {acc:.1f}%")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    from collections import defaultdict
    conf_matrix = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        conf_matrix[class_names[true]][class_names[pred]] += 1
    
    # Print confusion matrix
    header = "True\\Pred  " + " ".join([f"{c[:5]:>6s}" for c in class_names])
    print(header)
    for true_class in class_names:
        row = f"{true_class[:9]:9s} |"
        for pred_class in class_names:
            count = conf_matrix[true_class][pred_class]
            if count > 0:
                row += f" {count:5d}"
            else:
                row += "     ."
        print(row)
    
    # Convert to Float32 TFLite
    print("\n" + "=" * 60)
    print("CONVERTING TO FLOAT32 TFLITE")
    print("=" * 60)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # No quantization - pure float32
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(OUTPUT_DIR, 'wafer_classifier_float32.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_size = os.path.getsize(tflite_path) / 1024
    print(f"Float32 TFLite saved: {tflite_path}")
    print(f"Model size: {tflite_size:.1f} KB")
    
    # Generate C headers
    print("\nGenerating C headers...")
    
    # Labels header
    labels_h_path = os.path.join(OUTPUT_DIR, 'wafer_labels.h')
    with open(labels_h_path, 'w') as f:
        f.write("#ifndef WAFER_LABELS_H\n")
        f.write("#define WAFER_LABELS_H\n\n")
        f.write(f"#define NUM_CLASSES {num_classes}\n\n")
        f.write("const char* CLASS_NAMES[] = {\n")
        for name in class_names:
            f.write(f'    "{name}",\n')
        f.write("};\n\n")
        f.write("#endif  // WAFER_LABELS_H\n")
    
    # Model header
    model_h_path = os.path.join(OUTPUT_DIR, 'wafer_model.h')
    with open(model_h_path, 'w') as f:
        f.write("// Auto-generated Float32 TFLite model header\n")
        f.write(f"// Classes: {num_classes}\n")
        f.write("// Input: 128x128x1 float32\n\n")
        f.write("#ifndef WAFER_MODEL_H\n")
        f.write("#define WAFER_MODEL_H\n\n")
        f.write(f"const unsigned int wafer_model_len = {len(tflite_model)};\n")
        f.write("const unsigned char wafer_model[] = {\n")
        
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        
        f.write("\n};\n\n")
        f.write("#endif  // WAFER_MODEL_H\n")
    
    # Save config
    config = {
        'model': 'MobileNetV3Small',
        'input_shape': [1, IMG_SIZE[0], IMG_SIZE[1], 1],
        'num_classes': num_classes,
        'class_names': class_names,
        'dataset': DATASET_DIR,
        'train_images': len(train_images),
        'val_images': len(val_images),
        'test_images': len(test_images),
        'test_accuracy': float(test_accuracy),
        'tflite_size_kb': tflite_size,
        'format': 'float32'
    }
    
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {OUTPUT_DIR}")
    print(f"Classes: {num_classes}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"TFLite Size: {tflite_size:.1f} KB")

if __name__ == "__main__":
    main()
