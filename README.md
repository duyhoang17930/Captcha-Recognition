# Captcha Recognition

A deep learning-based captcha recognition system using CNN with CTC (Connectionist Temporal Classification) loss for sequence labeling.

## Overview

This project implements a neural network to recognize 5-character alphanumeric captcha codes from images. The model achieves ~100% validation accuracy.

## Dataset

- **Total samples**: 1,070 captcha images
- **Image size**: 200x50 pixels (RGBA)
- **Label length**: 5 characters
- **Character set**: 19 unique characters (2-8, b, c, d, e, f, g, m, n, p, w, x, y)
- **Train/Val split**: 963 / 107 (90%/10%)

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

## Results

| Metric | Value |
|--------|-------|
| Final Validation Accuracy | 100% |
| Best Model Epoch | 9 |
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
- `model_config.json` - Model configuration
- `data/samples/` - Training images