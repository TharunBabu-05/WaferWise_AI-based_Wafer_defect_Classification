"""
Data Augmentation Script for Hackathon_phase3_training_dataset
Augments 1000 images to ~4000 balanced images
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
import shutil

# Configuration
SOURCE_DIR = "Hackathon_phase3_training_dataset"
OUTPUT_DIR = "Hackathon_phase3_augmented_4000"
TARGET_IMAGES_PER_CLASS = 364  # ~4000 total for 11 classes
IMG_SIZE = (128, 128)

def augment_image(img, augmentation_type):
    """Apply a specific augmentation to an image."""
    
    if augmentation_type == 'flip_h':
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif augmentation_type == 'flip_v':
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    
    elif augmentation_type == 'rotate_90':
        return img.rotate(90)
    
    elif augmentation_type == 'rotate_180':
        return img.rotate(180)
    
    elif augmentation_type == 'rotate_270':
        return img.rotate(270)
    
    elif augmentation_type == 'rotate_small':
        angle = random.uniform(-15, 15)
        return img.rotate(angle, fillcolor=128)
    
    elif augmentation_type == 'brightness_up':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(random.uniform(1.1, 1.3))
    
    elif augmentation_type == 'brightness_down':
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(random.uniform(0.7, 0.9))
    
    elif augmentation_type == 'contrast_up':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(random.uniform(1.1, 1.4))
    
    elif augmentation_type == 'contrast_down':
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(random.uniform(0.6, 0.9))
    
    elif augmentation_type == 'sharpen':
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(random.uniform(1.5, 2.5))
    
    elif augmentation_type == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    elif augmentation_type == 'noise':
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    elif augmentation_type == 'zoom_in':
        # Crop center and resize back
        w, h = img.size
        crop_factor = random.uniform(0.8, 0.95)
        new_w, new_h = int(w * crop_factor), int(h * crop_factor)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BILINEAR)
    
    elif augmentation_type == 'shift':
        # Random shift
        shift_x = random.randint(-10, 10)
        shift_y = random.randint(-10, 10)
        return img.transform(img.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y), fillcolor=128)
    
    elif augmentation_type == 'combo1':
        # Flip + brightness
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(random.uniform(0.8, 1.2))
    
    elif augmentation_type == 'combo2':
        # Rotate + contrast
        img = img.rotate(random.uniform(-10, 10), fillcolor=128)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(random.uniform(0.8, 1.2))
    
    elif augmentation_type == 'combo3':
        # Flip + rotate 90
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img.rotate(90)
    
    return img

def main():
    print("=" * 60)
    print("DATA AUGMENTATION")
    print("=" * 60)
    
    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    # List of augmentation types
    augmentations = [
        'flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270',
        'rotate_small', 'brightness_up', 'brightness_down',
        'contrast_up', 'contrast_down', 'sharpen', 'blur', 'noise',
        'zoom_in', 'shift', 'combo1', 'combo2', 'combo3'
    ]
    
    # Get class directories
    class_dirs = sorted([d for d in os.listdir(SOURCE_DIR) 
                         if os.path.isdir(os.path.join(SOURCE_DIR, d))])
    
    print(f"\nFound {len(class_dirs)} classes")
    print(f"Target: {TARGET_IMAGES_PER_CLASS} images per class")
    print(f"Total target: ~{TARGET_IMAGES_PER_CLASS * len(class_dirs)} images")
    
    total_original = 0
    total_augmented = 0
    
    for class_name in class_dirs:
        source_class_dir = os.path.join(SOURCE_DIR, class_name)
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_dir)
        
        # Get all images in class
        image_files = [f for f in os.listdir(source_class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        original_count = len(image_files)
        total_original += original_count
        
        # Calculate how many augmentations needed
        augmentations_needed = TARGET_IMAGES_PER_CLASS - original_count
        
        print(f"\n{class_name}:")
        print(f"  Original: {original_count} images")
        
        # Copy original images first
        for img_file in image_files:
            src_path = os.path.join(source_class_dir, img_file)
            dst_path = os.path.join(output_class_dir, img_file)
            
            # Load, ensure grayscale and correct size, then save
            try:
                img = Image.open(src_path)
                if img.mode != 'L':
                    img = img.convert('L')
                if img.size != IMG_SIZE:
                    img = img.resize(IMG_SIZE, Image.BILINEAR)
                img.save(dst_path)
            except Exception as e:
                print(f"    Error copying {img_file}: {e}")
        
        # Generate augmented images
        aug_count = 0
        if augmentations_needed > 0:
            # Calculate augmentations per original image
            augs_per_image = augmentations_needed // original_count + 1
            
            for img_file in image_files:
                if aug_count >= augmentations_needed:
                    break
                    
                src_path = os.path.join(source_class_dir, img_file)
                
                try:
                    img = Image.open(src_path)
                    if img.mode != 'L':
                        img = img.convert('L')
                    if img.size != IMG_SIZE:
                        img = img.resize(IMG_SIZE, Image.BILINEAR)
                    
                    # Apply random augmentations
                    for i in range(augs_per_image):
                        if aug_count >= augmentations_needed:
                            break
                        
                        # Pick random augmentation
                        aug_type = random.choice(augmentations)
                        aug_img = augment_image(img.copy(), aug_type)
                        
                        # Ensure correct size
                        if aug_img.size != IMG_SIZE:
                            aug_img = aug_img.resize(IMG_SIZE, Image.BILINEAR)
                        
                        # Save augmented image
                        base_name = os.path.splitext(img_file)[0]
                        aug_filename = f"{base_name}_aug{aug_count}_{aug_type}.png"
                        aug_path = os.path.join(output_class_dir, aug_filename)
                        aug_img.save(aug_path)
                        aug_count += 1
                        
                except Exception as e:
                    print(f"    Error augmenting {img_file}: {e}")
        
        final_count = original_count + aug_count
        total_augmented += final_count
        print(f"  Augmented: {aug_count} new images")
        print(f"  Total: {final_count} images")
    
    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Original dataset: {total_original} images")
    print(f"Augmented dataset: {total_augmented} images")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
