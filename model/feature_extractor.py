import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import os
import pickle

# ---------- STEP 1: Load Pretrained CNN ----------
# Using the new weights format (no warnings now)
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove final classification layer
resnet.eval()  # set to evaluation mode (no training)

# ---------- STEP 2: Image Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # resize to standard CNN size
    transforms.ToTensor(),               # convert to tensor
    transforms.Normalize(                # normalize like ImageNet training
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- STEP 3: Feature Extraction Function ----------
def extract_features(image_path):
    """Extract a normalized feature vector from one image."""
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        features = resnet(img_t)
    features = features.squeeze().numpy()
    return features / np.linalg.norm(features)  # normalize vector

# ---------- STEP 4: Extract Features for All Images ----------
def extract_all_features(folder_path):
    """Extract features for all images inside a folder.
       Skips unreadable or corrupted images."""
    feature_dict = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder_path, filename)
            try:
                feature_vector = extract_features(path)
                feature_dict[filename] = feature_vector
                print(f"Extracted: {filename}")
            except Exception as e:
                print(f"⚠️ Skipped {filename}: {e}")
    return feature_dict

# ---------- STEP 5: Save to Database ----------
if __name__ == "__main__":
    folder = "static/images"
    features = extract_all_features(folder)

    with open("database/features.pkl", "wb") as f:
        pickle.dump(features, f)

    print("\n✅ All features saved to database/features.pkl")
