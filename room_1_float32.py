"""
MobileNetV3Small - Float32 Model with Grayscale Input
Target: i.MX RT1170-EVKB MCU
Input: 128x128x1 (grayscale, float32)
Output: float32

Same as mcu_model_12 but with float32 instead of INT8
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# Configuration
IMG_SIZE = 128
CHANNELS = 1  # Grayscale
BATCH_SIZE = 32
EPOCHS = 100
INITIAL_LR = 0.0005
PATIENCE = 20

DATA_DIR = Path("hackathon_balanced_2000")
OUTPUT_DIR = Path("model_output/mcu_model_13_float32")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MobileNetV3Small - Float32 Grayscale Model")
print("=" * 60)
print(f"Input: {IMG_SIZE}x{IMG_SIZE}x{CHANNELS} (Grayscale, float32)")
print(f"Output: float32")
print()

# Get class names
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# Save labels
with open(OUTPUT_DIR / "labels.json", "w") as f:
    json.dump(class_names, f, indent=2)
with open(OUTPUT_DIR / "labels.txt", "w") as f:
    f.write("\n".join(class_names))

# Load dataset as RGB first, then convert
def load_dataset():
    """Load images as RGB"""
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="int"
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        label_mode="int"
    )
    
    return train_ds, val_ds

train_ds, val_ds = load_dataset()

# Get all labels for class weights
all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(labels.numpy())
all_labels = np.array(all_labels)

class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights computed.")

# Convert RGB to grayscale
@tf.function
def rgb_to_grayscale(image, label):
    """Convert RGB to grayscale"""
    gray = tf.image.rgb_to_grayscale(image)  # [0, 255] range
    return gray, label

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# Build model - designed for grayscale with float32
def build_model():
    """
    Build a grayscale model with float32.
    Input range: [0, 255]
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), dtype=tf.float32, name="input")
    
    # Normalize to [-1, 1] for MobileNetV3
    x = layers.Rescaling(scale=1.0/127.5, offset=-1.0, name="normalize")(inputs)
    
    # Replicate grayscale to 3 channels for MobileNetV3
    x = layers.Concatenate(name="gray_to_rgb")([x, x, x])
    
    # MobileNetV3Small backbone with ImageNet weights
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,
        minimalistic=True,
        alpha=1.0
    )
    
    # Freeze backbone initially for transfer learning
    backbone.trainable = False
    
    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs, outputs, name="MobileNetV3Small_Grayscale_Float32")
    return model, backbone

model, backbone = build_model()
model.summary()

# ============================================================
# PHASE 1: Train with frozen backbone
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Training classifier (backbone frozen)")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Prepare datasets - convert to grayscale
train_ds_gray = train_ds.map(rgb_to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)
val_ds_gray = val_ds.map(rgb_to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)

# Apply augmentation
def augment(image, label):
    return data_augmentation(image, training=True), label

train_ds_aug = train_ds_gray.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds_aug = train_ds_aug.prefetch(tf.data.AUTOTUNE)
val_ds_prep = val_ds_gray.prefetch(tf.data.AUTOTUNE)

# Train phase 1
history1 = model.fit(
    train_ds_aug,
    validation_data=val_ds_prep,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ]
)

print(f"\nPhase 1 Best Val Accuracy: {max(history1.history['val_accuracy']):.4f}")

# ============================================================
# PHASE 2: Fine-tune entire model
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning entire model")
print("=" * 60)

# Unfreeze backbone
backbone.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(OUTPUT_DIR / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

history2 = model.fit(
    train_ds_aug,
    validation_data=val_ds_prep,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Save config
best_val_acc = max(history2.history['val_accuracy'])
config = {
    "model": "MobileNetV3Small",
    "input_size": [IMG_SIZE, IMG_SIZE, CHANNELS],
    "input_type": "grayscale",
    "input_dtype": "float32",
    "input_range": "[0, 255]",
    "num_classes": num_classes,
    "quantization": "none (float32)",
    "val_accuracy": float(best_val_acc)
}
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")

# ============================================================
# FLOAT32 TFLITE CONVERSION (No Quantization)
# ============================================================
print("\n" + "=" * 60)
print("FLOAT32 TFLITE CONVERSION (No Quantization)")
print("=" * 60)

# Reload best model
model = keras.models.load_model(str(OUTPUT_DIR / "best_model.keras"))

# Convert to TFLite WITHOUT quantization (float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No optimizations - keep as float32
# converter.optimizations = []  # Empty = no quantization

print("Converting to Float32 TFLite...")
tflite_model = converter.convert()

# Save TFLite model
tflite_path = OUTPUT_DIR / "wafer_classifier_float32.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"Saved: {tflite_path}")
print(f"Size: {size_mb:.2f} MB")

# Verify float32 model details
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"\nModel Input Details:")
print(f"  Name: {input_details['name']}")
print(f"  Shape: {input_details['shape']}")
print(f"  Type: {input_details['dtype']}")

print(f"\nModel Output Details:")
print(f"  Name: {output_details['name']}")
print(f"  Shape: {output_details['shape']}")
print(f"  Type: {output_details['dtype']}")

# ============================================================
# TEST ON ALL 2000 IMAGES
# ============================================================
print("\n" + "=" * 60)
print("FINAL TEST ON ALL 2000 IMAGES")
print("=" * 60)

# Load full dataset as grayscale
full_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    shuffle=False,
    color_mode="rgb",
    label_mode="int"
)

# Convert to grayscale
full_ds_gray = full_ds.map(rgb_to_grayscale, num_parallel_calls=tf.data.AUTOTUNE)

print("\nEvaluating Float32 TFLite on ALL data...")

correct = 0
total = 0
class_correct = {name: 0 for name in class_names}
class_total = {name: 0 for name in class_names}

for images, labels in full_ds_gray:
    # Images are [0, 255] float32
    img_float = images.numpy().astype(np.float32)
    
    # Set input directly (float32)
    interpreter.set_tensor(input_details['index'], img_float)
    interpreter.invoke()
    
    # Get float32 output
    output = interpreter.get_tensor(output_details['index'])[0]
    
    pred = np.argmax(output)
    label = labels.numpy()[0]
    
    class_name = class_names[label]
    class_total[class_name] += 1
    
    if pred == label:
        correct += 1
        class_correct[class_name] += 1
    total += 1

accuracy = correct / total * 100
print(f"\nFloat32 Grayscale Full Dataset Accuracy: {accuracy:.2f}%")

print("\nPer-Class:")
for name in class_names:
    if class_total[name] > 0:
        cls_acc = class_correct[name] / class_total[name] * 100
        print(f"  {name:12s}: {cls_acc:5.1f}% ({class_correct[name]}/{class_total[name]})")

# ============================================================
# GENERATE C HEADERS
# ============================================================
print("\nGenerating C headers...")

# Model header
model_bytes = tflite_model

with open(OUTPUT_DIR / "wafer_model.h", "w") as f:
    f.write("// Auto-generated TFLite model for MCU\n")
    f.write("// MobileNetV3Small - Float32 Grayscale\n")
    f.write(f"// Input: {IMG_SIZE}x{IMG_SIZE}x{CHANNELS} float32 grayscale [0, 255]\n")
    f.write(f"// Output: {num_classes} classes float32\n\n")
    f.write("#ifndef WAFER_MODEL_H\n")
    f.write("#define WAFER_MODEL_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define MODEL_INPUT_WIDTH {IMG_SIZE}\n")
    f.write(f"#define MODEL_INPUT_HEIGHT {IMG_SIZE}\n")
    f.write(f"#define MODEL_INPUT_CHANNELS {CHANNELS}\n")
    f.write(f"#define MODEL_NUM_CLASSES {num_classes}\n\n")
    f.write(f"const unsigned int wafer_model_len = {len(model_bytes)};\n")
    f.write("alignas(8) const unsigned char wafer_model[] = {\n")
    for i in range(0, len(model_bytes), 12):
        chunk = model_bytes[i:i+12]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        f.write(f"  {hex_str},\n")
    f.write("};\n\n")
    f.write("#endif // WAFER_MODEL_H\n")

# Labels header
with open(OUTPUT_DIR / "wafer_labels.h", "w") as f:
    f.write("// Auto-generated labels\n\n")
    f.write("#ifndef WAFER_LABELS_H\n")
    f.write("#define WAFER_LABELS_H\n\n")
    f.write(f"#define NUM_CLASSES {num_classes}\n\n")
    f.write("const char* const wafer_labels[] = {\n")
    for name in class_names:
        f.write(f'  "{name}",\n')
    f.write("};\n\n")
    f.write("#endif // WAFER_LABELS_H\n")

print("C headers generated.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
print(f"Model: {OUTPUT_DIR}")
print(f"Architecture: MobileNetV3Small")
print(f"Input: {IMG_SIZE}x{IMG_SIZE}x{CHANNELS} float32 (grayscale)")
print(f"Output: {num_classes} classes float32")
print(f"Val Accuracy: {best_val_acc*100:.2f}%")
print(f"Full Dataset (2000): {accuracy:.2f}%")
print(f"Size: {size_mb*1024:.1f} KB")

if accuracy >= 80:
    print("\n*** TARGET ACHIEVED: >80% ***")
else:
    print(f"\n*** Below target: {accuracy:.2f}% < 80% ***")
