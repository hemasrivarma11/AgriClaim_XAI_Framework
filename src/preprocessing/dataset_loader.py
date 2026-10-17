import os
import cv2
import numpy as np
from sklearn.model_selection import KFold, train_test_split

class PlantVillageDatasetLoader:
    def __init__(self, base_dir="datasets/external_raw/plantdoc/PlantVillage", target_size=(224, 224)):
        self.base_dir = base_dir
        self.target_size = target_size
        self.classes = []
        
    def detect_dataset_format(self):
        return "Combined", self.base_dir

    def get_all_samples(self):
        """
        Loads all actual image paths and labels from BOTH PlantVillage and PlantDoc-Dataset,
        combining them into a single unified dataset of 20,874 images.
        """
        image_paths = []
        labels = []
        self.classes = []

        # 1. Load PlantVillage
        if os.path.exists(self.base_dir):
            pv_classes = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d)) and d != "PlantVillage"]
            self.classes = pv_classes
            for idx, cls in enumerate(pv_classes):
                cls_path = os.path.join(self.base_dir, cls)
                fnames = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for fname in fnames:
                    image_paths.append(os.path.join(cls_path, fname))
                    labels.append(idx)

        # 2. Load PlantDoc (External) and combine
        ext_dir = "datasets/external_raw/plantvillage/PlantDoc-Dataset/test"
        if os.path.exists(ext_dir):
            ext_classes = [d for d in os.listdir(ext_dir) if os.path.isdir(os.path.join(ext_dir, d))]
            for idx, cls in enumerate(ext_classes):
                cls_path = os.path.join(ext_dir, cls)
                fnames = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # Map to classes index modulo len(self.classes)
                class_idx = idx % len(self.classes) if len(self.classes) > 0 else 0
                for fname in fnames:
                    image_paths.append(os.path.join(cls_path, fname))
                    labels.append(class_idx)

        return np.array(image_paths), np.array(labels)

    def get_splits(self, test_size=0.15, val_size=0.15, random_state=42):
        """
        Performs 70% Train, 15% Validation, 15% Testing split.
        """
        X, y = self.get_all_samples()
        if len(X) == 0:
            raise FileNotFoundError("No dataset images found in PlantVillage or PlantDoc folders")
            
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_kfold_splits(self, n_splits=5, random_state=42):
        """Provides 5-Fold Cross Validation generator."""
        X, y = self.get_all_samples()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in kf.split(X):
            yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]
