"""
Script untuk split dataset ke format Roboflow dengan rasio 7:2:1 (train:valid:test)
"""

import os
import shutil
import random
from pathlib import Path

# Konfigurasi
SOURCE_IMAGES = r"e:\Dataset\fire\fire_d-fire\datasets\train\images"
SOURCE_LABELS = r"e:\Dataset\fire\fire_d-fire\datasets\train\labels"
OUTPUT_DIR = r"e:\Dataset\fire\fire_d-fire\datasets\roboflow_split"

# Rasio split (train:valid:test = 7:2:1)
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# Seed untuk reproducibility
RANDOM_SEED = 42

def create_directory_structure(base_dir):
    """Membuat struktur folder untuk format Roboflow"""
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)
    print(f"[OK] Struktur folder dibuat di: {base_dir}")

def get_image_files(images_dir):
    """Mendapatkan semua file gambar"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for file in os.listdir(images_dir):
        if Path(file).suffix.lower() in valid_extensions:
            image_files.append(file)

    return sorted(image_files)

def split_dataset(image_files, train_ratio, valid_ratio, test_ratio):
    """Split dataset berdasarkan rasio yang ditentukan"""
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]

    return {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }

def copy_files(file_list, split_name, source_images, source_labels, output_dir):
    """Copy file gambar dan label ke folder tujuan"""
    copied_images = 0
    copied_labels = 0
    missing_labels = []

    dest_images = os.path.join(output_dir, split_name, 'images')
    dest_labels = os.path.join(output_dir, split_name, 'labels')

    for image_file in file_list:
        # Copy image
        src_image = os.path.join(source_images, image_file)
        dst_image = os.path.join(dest_images, image_file)
        shutil.copy2(src_image, dst_image)
        copied_images += 1

        # Copy label (dengan extensi .txt)
        label_file = Path(image_file).stem + '.txt'
        src_label = os.path.join(source_labels, label_file)
        dst_label = os.path.join(dest_labels, label_file)

        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
            copied_labels += 1
        else:
            missing_labels.append(label_file)

    print(f"  {split_name:6s}: {copied_images:4d} images, {copied_labels:4d} labels", end="")
    if missing_labels:
        print(f" ({len(missing_labels)} labels missing)")
    else:
        print()

    return copied_images, copied_labels, missing_labels

def create_data_yaml(output_dir, nc=1, names=['fire']):
    """Membuat file data.yaml untuk Roboflow/YOLO"""
    yaml_content = f"""# Roboflow Dataset Configuration
path: {output_dir}
train: train/images
val: valid/images
test: test/images

nc: {nc}
names: {names}
"""

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"[OK] File data.yaml dibuat di: {yaml_path}")

def main():
    print("="*60)
    print("SPLIT DATASET KE FORMAT ROBOFLOW (7:2:1)")
    print("="*60)
    print()

    # Validasi source directory
    if not os.path.exists(SOURCE_IMAGES):
        print(f"[ERROR] Folder images tidak ditemukan: {SOURCE_IMAGES}")
        return

    if not os.path.exists(SOURCE_LABELS):
        print(f"[ERROR] Folder labels tidak ditemukan: {SOURCE_LABELS}")
        return

    # Buat struktur folder
    print("1. Membuat struktur folder...")
    create_directory_structure(OUTPUT_DIR)
    print()

    # Dapatkan semua file gambar
    print("2. Membaca file gambar...")
    image_files = get_image_files(SOURCE_IMAGES)
    total_images = len(image_files)
    print(f"[OK] Total gambar ditemukan: {total_images}")
    print()

    # Split dataset
    print("3. Melakukan split dataset...")
    splits = split_dataset(image_files, TRAIN_RATIO, VALID_RATIO, TEST_RATIO)
    print(f"[OK] Train: {len(splits['train'])} ({len(splits['train'])/total_images*100:.1f}%)")
    print(f"[OK] Valid: {len(splits['valid'])} ({len(splits['valid'])/total_images*100:.1f}%)")
    print(f"[OK] Test:  {len(splits['test'])} ({len(splits['test'])/total_images*100:.1f}%)")
    print()

    # Copy files
    print("4. Menyalin file ke folder tujuan...")
    all_missing_labels = []

    for split_name, file_list in splits.items():
        _, _, missing = copy_files(
            file_list, split_name,
            SOURCE_IMAGES, SOURCE_LABELS,
            OUTPUT_DIR
        )
        all_missing_labels.extend(missing)

    print()

    # Buat data.yaml
    print("5. Membuat file konfigurasi...")
    create_data_yaml(OUTPUT_DIR)
    print()

    # Summary
    print("="*60)
    print("SELESAI!")
    print("="*60)
    print(f"Dataset berhasil di-split ke: {OUTPUT_DIR}")
    print()
    print("Struktur folder:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({len(splits['train'])} files)")
    print(f"  │   └── labels/ ({len(splits['train'])} files)")
    print(f"  ├── valid/")
    print(f"  │   ├── images/ ({len(splits['valid'])} files)")
    print(f"  │   └── labels/ ({len(splits['valid'])} files)")
    print(f"  ├── test/")
    print(f"  │   ├── images/ ({len(splits['test'])} files)")
    print(f"  │   └── labels/ ({len(splits['test'])} files)")
    print(f"  └── data.yaml")

    if all_missing_labels:
        print()
        print(f"[WARNING] {len(all_missing_labels)} label files tidak ditemukan")
        print("          (Gambar tetap di-copy, tapi tanpa label)")

if __name__ == "__main__":
    main()
