import os
import cv2
import glob
import random
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def balance_dataset(input_dir, output_dir, target_size=(48, 48)):
    """
    Balances the training dataset by oversampling minority classes with augmentation.
    NOTE: This is an offline process that creates a new, balanced dataset directory.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    class_counts = {}
    class_paths = {}

    for cls in os.listdir(input_dir):
        cls_path = os.path.join(input_dir, cls)
        if os.path.isdir(cls_path):
            image_files = glob.glob(os.path.join(cls_path, "*[.jpg|.jpeg|.png]"))
            if image_files:
                class_counts[cls] = len(image_files)
                class_paths[cls] = image_files

    if not class_counts:
        print("No image classes found in the input directory.")
        return

    print("Class distribution before balancing:", class_counts)
    max_count = max(class_counts.values())
    print(f"Largest class has {max_count} images. Augmenting others to match.")

    for cls, count in class_counts.items():
        output_class_dir = os.path.join(output_dir, cls)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        # Copy original images
        original_images = class_paths[cls]
        for img_file in original_images:
            shutil.copy(img_file, output_class_dir)

        # Augment missing images
        missing_count = max_count - count
        if missing_count > 0:
            print(f"Augmenting '{cls}': adding {missing_count} images ({count} -> {max_count})")
            
            for i in range(missing_count):
                img_path = random.choice(original_images)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                    
                img = cv2.resize(img, target_size)
                img = img.reshape((1,) + img.shape + (1,))

                # Generate and save one augmented image
                gen_flow = datagen.flow(img, batch_size=1,
                                        save_to_dir=output_class_dir,
                                        save_prefix="aug",
                                        save_format="jpg")
                next(gen_flow)

    print("\nBalancing complete! Final class distribution:")
    final_counts = {cls: len(os.listdir(os.path.join(output_dir, cls))) for cls in class_counts.keys()}
    print(final_counts)


def get_data_generators(train_dir, test_dir, batch_size=64):
    """
    Creates and returns the training and testing data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    print("Loading testing data...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False  # Keep data in order for evaluation
    )

    return train_generator, test_generator