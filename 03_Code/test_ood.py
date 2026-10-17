import sys
import os
import cv2
import numpy as np

sys.path.append('src')
from agents.vision_agent import CropVisionAgent

# Create dummy images to test
os.makedirs('scratch', exist_ok=True)

# 1. Non-crop image (pure blue/red image)
non_crop_img = np.zeros((224, 224, 3), dtype=np.uint8)
non_crop_img[:, :, 0] = 255 # Pure blue
non_crop_path = 'scratch/test_non_crop.png'
cv2.imwrite(non_crop_path, non_crop_img)

# 2. Simulated healthy leaf image (mostly green)
leaf_img = np.zeros((224, 224, 3), dtype=np.uint8)
leaf_img[:, :, 1] = 180 # Rich green
leaf_img[:, :, 0] = 30
leaf_img[:, :, 2] = 40
leaf_path = 'scratch/test_healthy_leaf.png'
cv2.imwrite(leaf_path, leaf_img)

agent = CropVisionAgent(use_pretrained=False)

print("=== Running OOD Image Validation Tests ===")
# Test non-crop image
r1 = agent.analyze(processed_dataset={"processed_image": np.zeros((224, 224, 3))}, image_path=non_crop_path)
print("Non-crop classification result:", r1["disease"])
print("Severity score:", r1["severity"])
print("Confidence:", r1["confidence"])
assert r1["disease"] == "Invalid/Non-Crop Image"

print("\n---")
# Test green leaf image
r2 = agent.analyze(processed_dataset={"processed_image": np.zeros((224, 224, 3))}, image_path=leaf_path)
print("Crop classification result:", r2["disease"])
print("Severity score:", r2["severity"])
print("Confidence:", r2["confidence"])
assert r2["disease"] != "Invalid/Non-Crop Image"

print("\nOOD Tests Completed Successfully!")
