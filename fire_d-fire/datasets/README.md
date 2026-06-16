# D-Fire Fire-Only Training Subset

## Overview
This is a fire-only training subset of the D-Fire dataset, containing only fire instances (original Class 1).
Smoke instances (original Class 0) have been removed from training data.

**Note:** Test set uses original D-Fire test directory for ground truth evaluation.

## Dataset Statistics

### Training Set (Fire-Only Subset)
- Total images: 4,707
- Total fire instances: 11,814
- Smoke instances removed: 9,550

### Test Set (Original D-Fire)
- Uses: E:/Dataset/D-Fire/test/
- Contains both Fire and Smoke classes
- For ground truth evaluation after training

## Class Mapping

### Training Subset
- **Original D-Fire:** Class 0 = Smoke, Class 1 = Fire
- **This Subset:** Class 0 = Fire (remapped from Class 1)

### Test Set (Original)
- Class 0 = Smoke
- Class 1 = Fire (for GT evaluation)

## Directory Structure
```
E:/Dataset/D-Fire/
├── subset-fire/
│   ├── train/
│   │   ├── images/   (fire instances only)
│   │   └── labels/   (Class 1 → Class 0)
│   ├── data.yaml
│   └── README.md
└── test/             (original, for GT evaluation)
    ├── images/
    └── labels/
```

## Usage

### Fine-tune on fire-only training data:
```bash
yolo train model=best.pt data=E:/Dataset/d-fire/subset-fire/data.yaml epochs=20 imgsz=640
```

### After training, test on original D-Fire test set:
The model will be evaluated on original test set containing both fire and smoke.

## Notes
- Training uses ONLY fire instances (Class 1)
- Test set remains original D-Fire for proper GT evaluation
- All training labels remapped: Class 1 → Class 0
- Images without fire instances excluded from training
- Original D-Fire: https://github.com/gaiasd/DFireDataset

Generated: 2026-01-01
