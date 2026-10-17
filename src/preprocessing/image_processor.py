import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        # ImageNet mean and std
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_image(self, path):
        """Load image as RGB numpy array."""
        image = Image.open(path).convert('RGB')
        return np.array(image)

    def resize(self, image):
        """Resize image to target size."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def normalize(self, image):
        """Normalize image pixel values in [0, 1] using ImageNet stats."""
        # Convert to float32 in [0, 1]
        img_float = image.astype(np.float32) / 255.0
        normalized = (img_float - self.mean) / self.std
        return normalized

    def extract_patches(self, image, patch_size=32):
        """
        Extract non-overlapping patches from the image.
        Returns a grid of patches for Swin Transformer patch embedding demonstration.
        """
        h, w, c = image.shape
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = image[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patches.append(patch)
        return patches

    def preprocess_pipeline(self, path):
        """Execute the full preprocessing pipeline on an image path."""
        raw_img = self.load_image(path)
        resized_img = self.resize(raw_img)
        normalized_img = self.normalize(resized_img)
        patches = self.extract_patches(resized_img)
        return {
            "raw": raw_img,
            "resized": resized_img,
            "normalized": normalized_img,
            "patches": patches
        }
