# Captcha Recognition

A deep learning-based captcha recognition system using CNN with CTC (Connectionist Temporal Classification) loss for sequence labeling.

## Overview

This project implements a neural network to recognize 5-character alphanumeric captcha codes from images. The model uses data augmentation and proper train/validation/test splits to ensure good generalization.

## Dataset

- **Total samples**: 1,070 captcha images
- **Image size**: 200x50 pixels (RGBA)
- **Label length**: 5 characters
- **Character set**: 19 unique characters (2-8, b, c, d, e, f, g, m, n, p, w, x, y)
- **Train/Val/Test split**: 856 / 107 / 107 (80% / 10% / 10%)

## Model Architecture

```
SimpleCNN + CTC
├── CNN Feature Extractor
│   ├── Conv2d(1, 64) + BatchNorm + ReLU
│   ├── Conv2d(64, 64) + BatchNorm + ReLU
│   ├── MaxPool2d(2,2)
│   ├── Conv2d(64, 128) + BatchNorm + ReLU
│   ├── Conv2d(128, 128) + BatchNorm + ReLU
│   ├── MaxPool2d(2,2)
│   ├── Conv2d(128, 256) + BatchNorm + ReLU
│   └── AdaptiveAvgPool2d((1, None))
├── FC Linear(256, num_classes)
└── CTC Loss (for sequence decoding)
```

Key insight: Using CNN only (without LSTM) avoids the CTC collapse issue and works well for fixed-length captcha sequences.

## Data Augmentation

Training uses the following augmentations:
- **RandomAffine**: rotate (±5°), translate (5%/10%), scale (0.9-1.1x), shear (±5°)
- **RandomErasing**: random noise patches (10% probability)

These augmentations help the model generalize better to unseen data.

## Results

| Metric | Value |
|--------|-------|
| Training Samples | 856 |
| Validation Samples | 107 |
| Test Samples (held-out) | 107 |
| Best Validation Accuracy | Saved in model_config.json |
| Best Test Accuracy (held-out) | Saved in model_config.json |
| Total Parameters | ~330K |

## Requirements

```
torch
torchvision
Pillow
numpy
matplotlib
scikit-learn
```

## Usage

### Training

Open `captcha_complete.ipynb` in Jupyter and run all cells.

### Prediction

```python
from PIL import Image
import torch
import torchvision.transforms as transforms
import json
import torch.nn as nn

# Define model
class SimpleCTC(nn.Module):
    def __init__(self, num_classes, num_channels=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(num_channels, num_channels*2, 3, padding=1), nn.BatchNorm2d(num_channels*2), nn.ReLU(),
            nn.Conv2d(num_channels*2, num_channels*2, 3, padding=1), nn.BatchNorm2d(num_channels*2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(num_channels*2, num_channels*4, 3, padding=1), nn.BatchNorm2d(num_channels*4), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.fc = nn.Linear(num_channels*4, num_classes)

    def forward(self, x):
        conv = self.cnn(x).squeeze(2).permute(0, 2, 1)
        return self.fc(conv).permute(1, 0, 2)

# CTC Decode function
def decode(outputs, idx_to_char, max_len=5):
    _, max_idx = torch.max(outputs, dim=2)
    max_idx = max_idx.permute(1, 0).cpu().numpy()
    decoded = []
    for seq in max_idx:
        chars, prev = [], -1
        for idx in seq:
            idx = int(idx)
            if idx != 0 and idx != prev:
                chars.append(idx_to_char.get(idx, ''))
            prev = idx
        decoded.append(''.join(chars[:max_len]))
    return decoded

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCTC(20)
model.load_state_dict(torch.load('best_ctc_model.pth', map_location=device))
model.eval()

# Load config
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
image = Image.open('captcha.png').convert('L')
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    pred = decode(outputs, config['idx_to_char'], 5)[0]

print(f"Predicted: {pred}")
```

## Files

- `captcha_complete.ipynb` - Complete training pipeline
- `best_ctc_model.pth` - Trained model weights
- `model_config.json` - Model configuration (includes best_val_acc and best_test_acc)
- `data/samples/` - Training images