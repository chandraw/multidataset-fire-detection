import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def find_all_images_labels(base_dirs):    
    image_label_pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        print(f"\n  Memproses: {base_dir}")
        
        all_images = []
        for ext in image_extensions:
            all_images.extend(base_path.rglob(f'*{ext}'))
            all_images.extend(base_path.rglob(f'*{ext.upper()}'))

        print(f"    Found {len(all_images)} images")

        matched_count = 0
        for img_file in all_images:            
            label_file = None

            if 'images' in img_file.parts:
                parts = list(img_file.parts)
                for i, part in enumerate(parts):
                    if part == 'images':
                        parts[i] = 'labels'
                        break
                label_path = Path(*parts).with_suffix('.txt')
                if label_path.exists():
                    label_file = label_path

            if not label_file:
                if img_file.parent.name == 'images':
                    label_dir = img_file.parent.parent / 'labels'
                    label_path = label_dir / (img_file.stem + '.txt')
                    if label_path.exists():
                        label_file = label_path

            if not label_file:
                possible_label_paths = [
                    img_file.with_suffix('.txt'),
                    img_file.parent.parent / 'labels' / img_file.parent.name / (img_file.stem + '.txt'),
                    base_path / 'labels' / (img_file.stem + '.txt'),
                ]

                for lbl_path in possible_label_paths:
                    if lbl_path.exists():
                        label_file = lbl_path
                        break

            if label_file and label_file.exists():
                image_label_pairs.append({
                    'image': img_file,
                    'label': label_file,
                    'source': base_dir
                })
                matched_count += 1

        print(f"    Successfully: {matched_count} images with labels")

    return image_label_pairs

def split_dataset(pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):    
    # Shuffle data
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs

def copy_files(pairs, output_dir, split_name):    
    img_dir = output_dir / split_name / 'images'
    lbl_dir = output_dir / split_name / 'labels'

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for idx, pair in enumerate(pairs):        
        source_prefix = Path(pair['source']).name.replace('-', '_')
        img_ext = pair['image'].suffix
        new_name = f"{source_prefix}_{idx:05d}"
        
        dest_img = img_dir / f"{new_name}{img_ext}"
        shutil.copy2(pair['image'], dest_img)

        dest_lbl = lbl_dir / f"{new_name}.txt"
        shutil.copy2(pair['label'], dest_lbl)

    print(f"  Copied {len(pairs)} files to {split_name}")

def create_data_yaml(output_dir, nc=1, class_names=None):    
    if class_names is None:
        class_names = ['fire']

    yaml_content = f"""# Fire Detection Dataset - Combined
# Combined from fire_d-fire, fire_fcmi, fire_roboflow
# Split ratio: 70% train, 20% val, 10% test

path: {output_dir.name}
train: train/images
val: val/images
test: test/images

nc: {nc}
names: {class_names}
"""

    yaml_file = output_dir / 'data.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated {yaml_file}")

def main():    
    random.seed(42)

    source_dirs = [
        'fire_d-fire',
        'fire_fcmi',
        'fire_roboflow'
    ]

    output_dir = Path('fire_mixed')

    print("=" * 60)
    print("Fire Dataset Merger")
    print("=" * 60)

    print("\nSearching images and labels from datasets...")
    all_pairs = find_all_images_labels(source_dirs)

    print(f"\n{'='*60}")
    print(f"Total pairs image-label found: {len(all_pairs)}")
    print(f"{'='*60}")

    source_stats = defaultdict(int)
    for pair in all_pairs:
        source_stats[pair['source']] += 1

    print("\nStatistic per dataset:")
    for source, count in source_stats.items():
        print(f"  {source}: {count} images")

    # Split dataset
    print(f"\nSplitting dataset with ratio 70:20:10...")
    train_pairs, val_pairs, test_pairs = split_dataset(all_pairs, 0.7, 0.2, 0.1)

    print(f"  Train: {len(train_pairs)} image ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} image ({len(val_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} image ({len(test_pairs)/len(all_pairs)*100:.1f}%)")
    
    if output_dir.exists():
        print(f"\nRemoving old folder: {output_dir}")
        shutil.rmtree(output_dir)

    # Copy files
    print("\nCopying files to output folder...")
    copy_files(train_pairs, output_dir, 'train')
    copy_files(val_pairs, output_dir, 'val')
    copy_files(test_pairs, output_dir, 'test')

    # Creating data.yaml
    print("\nCreating data.yaml...")
    create_data_yaml(output_dir, nc=1, class_names=['fire'])

    print("\n" + "=" * 60)
    print("Finished!")
    print(f"Mixed Dataset saved at: {output_dir.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
