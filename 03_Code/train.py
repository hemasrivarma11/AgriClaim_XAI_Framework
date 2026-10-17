import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import timm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("timm or albumentations not found. Run: pip install timm albumentations")

# Define Multi-Task Swin Transformer Network
class MultiTaskSwin(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(MultiTaskSwin, self).__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        in_features = self.backbone.head.in_features
        
        # Remove default classifier
        self.backbone.head = nn.Identity()
        
        # Multi-task heads
        self.cls_head = nn.Linear(in_features, num_classes) # Disease category classification
        self.sev_head = nn.Linear(in_features, 1)          # Severity score regression

    def forward(self, x):
        features = self.backbone.forward_features(x)
        # Global average pooling (Swin output shape is usually B, L, C or B, C, H, W depending on version)
        if len(features.shape) == 3:
            # Shape is B, L, C -> pool over sequence dimension
            features = features.mean(dim=1)
        elif len(features.shape) == 4:
            # Detect channel configuration (channels-last vs channels-first)
            if features.shape[-1] == self.cls_head.in_features:
                features = features.mean(dim=[1, 2]) # Pool over H, W
            else:
                features = features.mean(dim=[2, 3]) # Pool over H, W
            
        cls_out = self.cls_head(features)
        sev_out = torch.sigmoid(self.sev_head(features)) # Map severity to [0, 1]
        
        return cls_out, sev_out.squeeze(-1)

# Custom Multi-Task Dataset
class CropDiseaseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        df must contain columns: 'image_name', 'class_idx', 'severity'
        """
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_name'])
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            # Fallback random image if file reading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        class_idx = int(row['class_idx'])
        severity = float(row['severity'])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Standard tensor convert
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, class_idx, severity

# Helper to generate synthetic data for dry run validation
def generate_synthetic_dataset(data_dir, num_samples=60):
    print("Generating synthetic crop leaf images for dry-run training...")
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    classes = ["Healthy", "Leaf Blight", "Pest Infestation", "Leaf Rust"]
    records = []
    
    for i in range(num_samples):
        class_idx = np.random.choice([0, 1, 2, 3])
        class_name = classes[class_idx]
        severity = 0.0 if class_name == "Healthy" else float(np.random.uniform(0.15, 0.90))
        img_name = f"synth_{i:04d}_{class_name.lower().replace(' ', '_')}.png"
        
        # Generate dummy color leaf image
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        # Base green leaf color
        img_arr[:, :] = [34, 139, 34] 
        
        if class_name != "Healthy":
            # Add spots/lesions (brown/yellow spots)
            num_spots = int(severity * 10) + 1
            for _ in range(num_spots):
                cx, cy = np.random.randint(20, 204, size=2)
                r = np.random.randint(5, 20)
                # Draw brown spot
                color = [139, 69, 19] if class_name == "Leaf Blight" else [184, 134, 11]
                img_arr = cv2.circle(img_arr, (cx, cy), r, color, -1)
                
        # Save image
        cv2.imwrite(os.path.join(raw_dir, img_name), cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR))
        
        records.append({
            "image_name": img_name,
            "class_name": class_name,
            "class_idx": class_idx,
            "severity": severity
        })
        
    df = pd.DataFrame(records)
    metadata_dir = os.path.join(data_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    df.to_csv(os.path.join(metadata_dir, "annotations.csv"), index=False)
    print(f"Generated annotations.csv with {num_samples} records.")
    return df, raw_dir

# Training loop
def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check dataset
    annotations_path = os.path.join(args.data_dir, "metadata", "annotations.csv")
    img_dir = os.path.join(args.data_dir, "raw")
    
    if not os.path.exists(annotations_path) or args.dry_run:
        df, img_dir = generate_synthetic_dataset(args.data_dir)
    else:
        df = pd.read_csv(annotations_path)
        
    # Split into train/validation
    np.random.seed(42)
    msk = np.random.rand(len(df)) < 0.8
    train_df = df[msk].reset_index(drop=True)
    val_df = df[~msk].reset_index(drop=True)
    print(f"Dataset split: Train samples = {len(train_df)} | Val samples = {len(val_df)}")

    # Define Data Augmentation using Albumentations
    if ALBUMENTATIONS_AVAILABLE:
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        train_transform = None
        val_transform = None

    # Load datasets
    train_dataset = CropDiseaseDataset(train_df, img_dir, train_transform)
    val_dataset = CropDiseaseDataset(val_df, img_dir, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Initialize model
    print("Loading Swin Transformer...")
    model = MultiTaskSwin(num_classes=args.num_classes, pretrained=True)
    model.to(device)

    # Loss Functions and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_sev = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    os.makedirs(args.model_dir, exist_ok=True)

    print("Beginning Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_sev_loss = 0.0
        
        start_time = time.time()
        
        for images, class_idxs, severities in train_loader:
            images = images.to(device)
            class_idxs = class_idxs.to(device)
            severities = severities.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass
            cls_out, sev_out = model(images)
            
            # Joint Loss calculation
            loss_cls = criterion_cls(cls_out, class_idxs)
            loss_sev = criterion_sev(sev_out, severities)
            loss = loss_cls + (args.lambda_sev * loss_sev)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_cls_loss += loss_cls.item() * images.size(0)
            train_sev_loss += loss_sev.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        train_cls_loss /= len(train_dataset)
        train_sev_loss /= len(train_dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_sev_loss = 0.0
        correct_cls = 0
        
        with torch.no_grad():
            for images, class_idxs, severities in val_loader:
                images = images.to(device)
                class_idxs = class_idxs.to(device)
                severities = severities.to(device, dtype=torch.float32)
                
                cls_out, sev_out = model(images)
                
                loss_cls = criterion_cls(cls_out, class_idxs)
                loss_sev = criterion_sev(sev_out, severities)
                loss = loss_cls + (args.lambda_sev * loss_sev)
                
                val_loss += loss.item() * images.size(0)
                val_cls_loss += loss_cls.item() * images.size(0)
                val_sev_loss += loss_sev.item() * images.size(0)
                
                preds = cls_out.argmax(dim=1)
                correct_cls += (preds == class_idxs).sum().item()
                
        val_loss /= len(val_dataset)
        val_cls_loss /= len(val_dataset)
        val_sev_loss /= len(val_dataset)
        val_acc = (correct_cls / len(val_dataset)) * 100
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} [{elapsed:.1f}s] -> "
              f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.3f}, Sev: {train_sev_loss:.3f}) | "
              f"Val Loss: {val_loss:.4f} (Cls: {val_cls_loss:.3f}, Sev: {val_sev_loss:.3f}) | "
              f"Val Cls Acc: {val_acc:.1f}%")
              
        # Checkpoint saving
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = os.path.join(args.model_dir, "swin_crop_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  --> Saved new best model checkpoint to {model_path}")
            
    print("Multi-task Swin Transformer training completed successfully!")

if __name__ == "__main__":
    import cv2 # Required for synthetic generation
    parser = argparse.ArgumentParser(description="Multi-Task Swin Transformer Training Script")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Path to dataset directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lambda_sev", type=float, default=1.0, help="Balancer weight for severity loss head")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of crop damage categories")
    parser.add_argument("--dry_run", action="store_true", default=True, help="Force dry-run using simulated dataset splits")

    args = parser.parse_args()
    train_model(args)
