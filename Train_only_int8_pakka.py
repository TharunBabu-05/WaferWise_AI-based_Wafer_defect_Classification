"""
MobileNetV3Small - Pure INT8 Quantized Model with Grayscale Input
Target: i.MX RT1170-EVKB MCU
Input: 128x128x1 (grayscale, INT8)
Output: INT8

Key fixes:
1. Use pretrained weights and properly adapt grayscale
2. Correct quantization-aware preprocessing
3. Better training strategy
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
OUTPUT_DIR = Path("model_output/mcu_model_12")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MobileNetV3Small - Pure INT8 Grayscale Model v2")
print("=" * 60)
print(f"Input: {IMG_SIZE}x{IMG_SIZE}x{CHANNELS} (Grayscale)")
print(f"Quantization: Full INT8 (input, weights, output)")
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
# This ensures we can use pretrained weights
def load_dataset():
    """Load images as RGB"""
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",  # Load as RGB
        label_mode="int"
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="rgb",  # Load as RGB
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

# Convert RGB to grayscale (average method)
@tf.function
def rgb_to_grayscale(image, label):
    """Convert RGB to grayscale using luminosity method"""
    gray = tf.image.rgb_to_grayscale(image)  # [0, 255] range
    return gray, label

# Data augmentation (light)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

# Build model - designed for grayscale with INT8 quantization in mind
def build_model():
    """
    Build a quantization-friendly grayscale model.
    Input range: [0, 255] (will be quantized to INT8)
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), dtype=tf.float32, name="input")
    
    # Normalize to [-1, 1] for MobileNetV3
    # INT8 range is [-128, 127], so [0, 255] -> [-128, 127] with scale ~1.0
    # But for training we use float, then quantize
    x = layers.Rescaling(scale=1.0/127.5, offset=-1.0, name="normalize")(inputs)
    
    # Replicate grayscale to 3 channels for MobileNetV3
    x = layers.Concatenate(name="gray_to_rgb")([x, x, x])
    
    # MobileNetV3Small backbone with ImageNet weights
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",  # Use pretrained weights!
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
    
    model = keras.Model(inputs, outputs, name="MobileNetV3Small_Grayscale_INT8")
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
    "input_range": "[0, 255]",
    "num_classes": num_classes,
    "quantization": "full_int8",
    "val_accuracy": float(best_val_acc)
}
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")

# ============================================================
# PURE INT8 QUANTIZATION
# ============================================================
print("\n" + "=" * 60)
print("PURE INT8 QUANTIZATION")
print("=" * 60)

# Reload best model
model = keras.models.load_model(str(OUTPUT_DIR / "best_model.keras"))

# Representative dataset - use grayscale images in [0, 255] range
def representative_dataset_gen():
    """Generate representative dataset for INT8 calibration"""
    count = 0
    for images, _ in val_ds_gray.unbatch():
        if count >= 200:  # Use 200 samples for calibration
            break
        # Image is [H, W, 1] in [0, 255] range
        img = tf.expand_dims(images, 0)  # Add batch dim
        img = tf.cast(img, tf.float32)
        yield [img]
        count += 1

# Convert to TFLite with FULL INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for calibration
converter.representative_dataset = representative_dataset_gen

# Force INT8 input and output
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8   # Input: INT8
converter.inference_output_type = tf.int8  # Output: INT8

print("Converting to Pure INT8 TFLite...")
try:
    tflite_model = converter.convert()
except Exception as e:
    print(f"INT8 conversion failed: {e}")
    print("Trying with UINT8 input instead...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8   # Input: UINT8
    converter.inference_output_type = tf.int8   # Output: INT8
    
    tflite_model = converter.convert()

# Save TFLite model
tflite_path = OUTPUT_DIR / "wafer_classifier_int8.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_mb = len(tflite_model) / (1024 * 1024)
print(f"Saved: {tflite_path}")
print(f"Size: {size_mb:.2f} MB")

# Verify INT8 model details
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"\nModel Input Details:")
print(f"  Name: {input_details['name']}")
print(f"  Shape: {input_details['shape']}")
print(f"  Type: {input_details['dtype']}")
if 'quantization_parameters' in input_details:
    quant = input_details['quantization_parameters']
    if len(quant['scales']) > 0:
        print(f"  Scale: {quant['scales'][0]}")
        print(f"  Zero Point: {quant['zero_points'][0]}")

print(f"\nModel Output Details:")
print(f"  Name: {output_details['name']}")
print(f"  Shape: {output_details['shape']}")
print(f"  Type: {output_details['dtype']}")
if 'quantization_parameters' in output_details:
    quant = output_details['quantization_parameters']
    if len(quant['scales']) > 0:
        print(f"  Scale: {quant['scales'][0]}")
        print(f"  Zero Point: {quant['zero_points'][0]}")

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

print("\nEvaluating Pure INT8 TFLite on ALL data...")

# Get quantization parameters
input_scale = input_details['quantization_parameters']['scales'][0]
input_zero_point = input_details['quantization_parameters']['zero_points'][0]
output_scale = output_details['quantization_parameters']['scales'][0]
output_zero_point = output_details['quantization_parameters']['zero_points'][0]
input_dtype = input_details['dtype']

print(f"Input dtype: {input_dtype}")
print(f"Input scale: {input_scale}, zero_point: {input_zero_point}")

correct = 0
total = 0
class_correct = {name: 0 for name in class_names}
class_total = {name: 0 for name in class_names}

for images, labels in full_ds_gray:
    # Images are [0, 255] float32
    img_float = images.numpy()[0]
    
    # Quantize to INT8 or UINT8
    if input_dtype == np.int8:
        img_quant = np.clip(np.round(img_float / input_scale + input_zero_point), -128, 127).astype(np.int8)
    else:  # uint8
        img_quant = np.clip(np.round(img_float / input_scale + input_zero_point), 0, 255).astype(np.uint8)
    
    img_quant = np.expand_dims(img_quant, axis=0)
    
    interpreter.set_tensor(input_details['index'], img_quant)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details['index'])[0]
    
    # Dequantize if needed
    if output_details['dtype'] != np.float32:
        output_float = (output.astype(np.float32) - output_zero_point) * output_scale
    else:
        output_float = output
    
    pred = np.argmax(output_float)
    label = labels.numpy()[0]
    
    class_name = class_names[label]
    class_total[class_name] += 1
    
    if pred == label:
        correct += 1
        class_correct[class_name] += 1
    total += 1

accuracy = correct / total * 100
print(f"\nPure INT8 Grayscale Full Dataset Accuracy: {accuracy:.2f}%")

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
input_type_str = "int8_t" if input_dtype == np.int8 else "uint8_t"

with open(OUTPUT_DIR / "wafer_model.h", "w") as f:
    f.write("// Auto-generated TFLite model for MCU\n")
    f.write("// MobileNetV3Small - Pure INT8 Grayscale\n")
    f.write(f"// Input: {IMG_SIZE}x{IMG_SIZE}x{CHANNELS} {input_type_str} grayscale\n")
    f.write(f"// Output: {num_classes} classes int8_t\n\n")
    f.write("#ifndef WAFER_MODEL_H\n")
    f.write("#define WAFER_MODEL_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define MODEL_INPUT_WIDTH {IMG_SIZE}\n")
    f.write(f"#define MODEL_INPUT_HEIGHT {IMG_SIZE}\n")
    f.write(f"#define MODEL_INPUT_CHANNELS {CHANNELS}\n")
    f.write(f"#define MODEL_NUM_CLASSES {num_classes}\n")
    f.write(f"#define MODEL_INPUT_SCALE {input_scale}f\n")
    f.write(f"#define MODEL_INPUT_ZERO_POINT {input_zero_point}\n")
    f.write(f"#define MODEL_OUTPUT_SCALE {output_scale}f\n")
    f.write(f"#define MODEL_OUTPUT_ZERO_POINT {output_zero_point}\n\n")
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
print(f"Input: {IMG_SIZE}x{IMG_SIZE}x{CHANNELS} {input_type_str} (grayscale)")
print(f"Output: {num_classes} classes int8")
print(f"Val Accuracy: {best_val_acc*100:.2f}%")
print(f"Full Dataset (2000): {accuracy:.2f}%")
print(f"Size: {size_mb*1024:.1f} KB")

if accuracy >= 80:
    print("\n*** TARGET ACHIEVED: >80% ***")
else:
    print(f"\n*** Below target: {accuracy:.2f}% < 80% ***")
