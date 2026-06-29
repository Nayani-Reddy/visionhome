import os
import cv2
import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
ALLOWED_EXT = [".jpg", ".jpeg", ".png"]

def get_dominant_colors(img_path, k=3):
    img = cv2.imread(str(img_path))
    if img is None:
        return [], "modern"
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate brightness & contrast
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    pixels = img_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # Subsample for speed
    if len(pixels) > 10000:
        pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]
        
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    # Style heuristics
    if brightness > 180 or (brightness > 150 and contrast < 50):
        style = "minimal"
    elif brightness < 100 or contrast > 75:
        style = "luxury"
    else:
        style = "modern"
        
    return colors.tolist(), style

def build_features():
    print("Loading ResNet50 model for feature extraction...")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    dataset_metadata = {}
    
    for root, dirs, files in os.walk(DATASETS_DIR):
        for f in files:
            if Path(f).suffix.lower() not in ALLOWED_EXT:
                continue
                
            abs_path = Path(root) / f
            rel_path = str(abs_path.relative_to(DATASETS_DIR))
            print(f"[AI Pipeline] Processing: {rel_path}")
            
            try:
                # 1. Feature extraction
                img_load = image.load_img(abs_path, target_size=(224, 224))
                x = image.img_to_array(img_load)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                features = model.predict(x, verbose=0)[0]
                features_norm = features / (np.linalg.norm(features) + 1e-10)
                
                # 2. Extract colors and style
                dom_colors, style = get_dominant_colors(abs_path)
                
                dataset_metadata[rel_path] = {
                    "features": features_norm,
                    "dominant_colors": dom_colors,
                    "style": style
                }
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
                
    output_path = PROJECT_ROOT / "dataset_metadata.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(dataset_metadata, f)
        
    print(f"Successfully configured AI metadata for {len(dataset_metadata)} items. Saved to {output_path}")

if __name__ == "__main__":
    build_features()
