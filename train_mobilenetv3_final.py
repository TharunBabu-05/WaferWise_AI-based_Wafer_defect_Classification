"""
MCU Model Training - MobileNetV3Small
Target: >80% accuracy with INT8 quantization
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
from datetime import datetime
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

CONFIG = {
    'img_size': 128,
    'batch_size': 32,
    'epochs': 80,
    'initial_lr': 0.0005,
    'min_lr': 1e-6,
    'patience': 15,
    'val_split': 0.2,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'hackathon_balanced_2000')
OUTPUT_BASE = os.path.join(BASE_DIR, 'model_output')


def get_next_model_number():
    existing = [d for d in os.listdir(OUTPUT_BASE) if d.startswith('mcu_model_')]
    if not existing:
        return 1
    numbers = [int(d.split('_')[-1]) for d in existing if d.split('_')[-1].isdigit()]
    return max(numbers) + 1 if numbers else 1


def load_all_data():
    print("\nLoading data...")
    class_names = sorted(os.listdir(DATA_DIR))
    print(f"Classes: {class_names}")
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((CONFIG['img_size'], CONFIG['img_size']))
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error: {img_path}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"Loaded {len(images)} images, shape: {images.shape}")
    return images, labels, class_names


def create_model(num_classes):
    """Create MobileNetV3Small model."""
    
    inputs = keras.Input(shape=(CONFIG['img_size'], CONFIG['img_size'], 3))
    
    # Light augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.08)(x)
    
    # Scale from [0,1] to [-1,1] for MobileNetV3
    x = layers.Rescaling(scale=2.0, offset=-1.0)(x)
    
    # MobileNetV3Small backbone
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(CONFIG['img_size'], CONFIG['img_size'], 3),
        alpha=1.0,
        minimalistic=True,  # Better for quantization
        include_top=False,
        weights='imagenet',
        include_preprocessing=False  # We handle preprocessing ourselves
    )
    backbone.trainable = True
    
    x = backbone(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def convert_to_tflite(model, X_cal, output_path):
    print("\nConverting to INT8 TFLite...")
    
    def representative_dataset():
        indices = np.random.choice(len(X_cal), min(300, len(X_cal)), replace=False)
        for i in indices:
            yield [X_cal[i:i+1].astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Saved: {output_path}")
    print(f"Size: {len(tflite_model)/1024/1024:.2f} MB")
    return tflite_model


def evaluate_tflite_full(tflite_path, X, y, class_names):
    print("\nEvaluating TFLite on ALL data...")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    
    correct = 0
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    for i in range(len(X)):
        interpreter.set_tensor(inp[0]['index'], X[i:i+1].astype(np.float32))
        interpreter.invoke()
        pred = np.argmax(interpreter.get_tensor(out[0]['index'])[0])
        
        class_total[y[i]] += 1
        if pred == y[i]:
            correct += 1
            class_correct[y[i]] += 1
    
    accuracy = correct / len(X)
    print(f"\nTFLite INT8 Full Dataset Accuracy: {accuracy*100:.2f}%")
    
    print("\nPer-Class:")
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {name:12s}: {acc*100:5.1f}% ({class_correct[i]}/{class_total[i]})")
    
    return accuracy


def generate_headers(model_data, class_names, output_dir):
    header = f'''// MobileNetV3Small TFLite Model - Generated {datetime.now()}
#ifndef WAFER_MODEL_H
#define WAFER_MODEL_H

const unsigned int wafer_model_len = {len(model_data)};
alignas(16) const unsigned char wafer_model[] = {{
'''
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        header += '    ' + ', '.join(f'0x{b:02X}' for b in chunk) + ',\n'
    header += '};\n#endif\n'
    
    with open(os.path.join(output_dir, 'wafer_model.h'), 'w') as f:
        f.write(header)
    
    labels = '#ifndef WAFER_LABELS_H\n#define WAFER_LABELS_H\n\n'
    labels += f'#define NUM_CLASSES {len(class_names)}\n\n'
    labels += 'const char* const CLASS_LABELS[] = {\n'
    for name in class_names:
        labels += f'    "{name}",\n'
    labels += '};\n#endif\n'
    
    with open(os.path.join(output_dir, 'wafer_labels.h'), 'w') as f:
        f.write(labels)
    
    print("C headers generated.")


def main():
    print("="*60)
    print("MobileNetV3Small Training - Target: 80%+ Accuracy")
    print("="*60)
    
    model_num = get_next_model_number()
    output_dir = os.path.join(OUTPUT_BASE, f'mcu_model_{model_num}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    X, y, class_names = load_all_data()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=CONFIG['val_split'], stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))
    
    model = create_model(len(class_names))
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['initial_lr']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['patience'],
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=6,
            min_lr=CONFIG['min_lr'],
            mode='max',
            verbose=1
        )
    ]
    
    print("\nTraining MobileNetV3Small...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")
    
    tflite_path = os.path.join(output_dir, 'wafer_classifier_int8.tflite')
    tflite_model = convert_to_tflite(model, X_train, tflite_path)
    
    print("\n" + "="*60)
    print("FINAL TEST ON ALL 2000 IMAGES")
    print("="*60)
    full_acc = evaluate_tflite_full(tflite_path, X, y, class_names)
    
    generate_headers(tflite_model, class_names, output_dir)
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump({
            **CONFIG, 
            'backbone': 'MobileNetV3Small',
            'best_val_acc': float(best_val_acc), 
            'full_acc': float(full_acc)
        }, f, indent=2)
    
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(class_names, f)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Model: {output_dir}")
    print(f"Backbone: MobileNetV3Small")
    print(f"Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"Full Dataset (2000): {full_acc*100:.2f}%")
    print(f"Size: {len(tflite_model)/1024:.1f} KB")
    
    if full_acc >= 0.80:
        print("\n*** TARGET ACHIEVED: >80% ***")
    else:
        print(f"\nGap to 80%: {(0.80-full_acc)*100:.1f}%")


if __name__ == '__main__':
    main()
