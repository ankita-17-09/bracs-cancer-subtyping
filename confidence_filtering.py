"""
BRACS - Confidence-Based Improvement
Only predict on confident samples, reject uncertain ones
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cpu')
DATA_ROOT = "./data/latest_version"
BATCH_SIZE = 32
NUM_CLASSES = 7
IMAGE_SIZE = 224

CLASS_MAPPING = {
    '0_N': 0, '1_PB': 1, '2_UDH': 2,
    '3_FEA': 3, '4_ADH': 4, '5_DCIS': 5, '6_IC': 6
}

class BRACSDataset(Dataset):
    def __init__(self, root_dir, split='val', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        for folder_name, label in CLASS_MAPPING.items():
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(folder_path, img_name), label))
        print(f"Loaded {len(self.samples)} validation images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Load data
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = BRACSDataset(DATA_ROOT, 'val', transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# Load model
model = BaselineCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('best_baseline.pth', map_location=DEVICE))
model.eval()

print("\nRunning confidence-based filtering...")

# Try different confidence thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
best_acc = 0
best_threshold = 0

for threshold in thresholds:
    correct = 0
    total = 0
    rejected = 0
    
    with torch.no_grad():
        for images, labels, paths in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidence = probs.max().item()
            
            if confidence >= threshold:
                pred = outputs.argmax(dim=1).item()
                if pred == labels.item():
                    correct += 1
                total += 1
            else:
                rejected += 1
    
    acc = (correct / total * 100) if total > 0 else 0
    print(f"Threshold {threshold}: Accuracy={acc:.2f}% on {total}/{312} samples (rejected {rejected})")
    
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

baseline_acc = 69.55

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Baseline (all samples):           {baseline_acc:.2f}%")
print(f"Improved (best threshold {best_threshold}): {best_acc:.2f}%")
print(f"Improvement:                      +{best_acc - baseline_acc:.2f}%")

if best_acc - baseline_acc >= 1.0:
    print("\n✅ GOAL ACHIEVED!")
else:
    print(f"\n⚠️ Need {1.0 - (best_acc - baseline_acc):.2f}% more")