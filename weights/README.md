# Weight Files

This directory contains the backbone weight files for YOLACT Edge training.

## Required Files

The following weight files are required for training with different backbones:

### ResNet50 Backbone
- **File**: `resnet50-19c8e357-fixed.pth`
- **Size**: ~121 MB
- **Config**: Used with `yolact_edge_resnet50_config`

### ResNet101 Backbone  
- **File**: `resnet101_reducedfc.pth`
- **Size**: ~194 MB
- **Config**: Used with `yolact_edge_config` (default)

### MobileNetV2 Backbone
- **File**: `mobilenet_v2-b0353104.pth`
- **Size**: ~194 MB
- **Config**: Used with `yolact_edge_mobilenetv2_config`

## Download Instructions

These weight files are not included in the repository due to their large size (>100MB). You need to download them separately:

### Option 1: Download from Official Sources
```bash
# ResNet50
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O weights/resnet50-19c8e357-fixed.pth

# ResNet101  
wget https://download.pytorch.org/models/resnet101_reducedfc.pth -O weights/resnet101_reducedfc.pth

# MobileNetV2
wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth -O weights/mobilenet_v2-b0353104.pth
```

### Option 2: Manual Download
1. Download the files from the official PyTorch model zoo
2. Place them in the `weights/` directory with the exact names listed above
3. Ensure the files have the correct permissions

## Verification

You can verify that the weight files are properly loaded by running:

```python
import torch
checkpoint = torch.load('weights/resnet50-19c8e357-fixed.pth', map_location='cpu')
print(f"Loaded checkpoint with {len(checkpoint)} keys")
```

## Notes

- The weight files are large (>100MB) and are excluded from version control
- Make sure to use the exact filenames as specified in the config files
- These are pre-trained ImageNet weights that will be fine-tuned during training 