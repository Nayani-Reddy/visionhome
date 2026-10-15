"""
VisionHome Machine Learning Feature Extraction & Dataset Precomputation Script
=============================================================================

This script processes the furniture image dataset and precomputes multi-parameter
visual feature representations for the Machine Learning (ML) recommendation system.

ML Architectural Classification:
--------------------------------
1. DEEP LEARNING (DL):
   - Pretrained ResNet50 CNN (ImageNet weights, global average pooling).
   - Extracts 2048-dimensional visual feature vectors for visual similarity matching.

2. UNSUPERVISED MACHINE LEARNING:
   - K-Means Clustering (k=3 centroids) on RGB pixel color samples.
   - Extracts dominant color palette centroids for palette compatibility analysis.

3. COMPUTER VISION & IMAGE-PROCESSING FEATURE EXTRACTION:
   - Color Distribution: Normalized 2D HSV color histogram (8x8 bins for H & S).
   - Brightness: Normalized mean grayscale intensity (0.0 to 1.0).
   - Micro-Texture Descriptor: Local Binary Pattern (LBP, P=8, R=1, 10-bin uniform histogram).
   - Shape/Edge Structure: Canny edge detector density & 8-bin edge orientation histogram.
   - Visual Style Feature Vector: Vector of brightness, contrast, saturation, and texture variance.

4. METADATA & CATEGORY RELEVANCE:
   - Furniture category tag parsed from dataset directory path structure.
"""

import os
import sys
import time
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
METADATA_FILE = PROJECT_ROOT / "dataset_metadata.pkl"
ALLOWED_EXT = [".jpg", ".jpeg", ".png"]

try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow / Keras not available.")

def extract_cv_features(img_bgr):
    """
    Extracts classical computer vision & image processing features from a BGR image.
    Returns: dominant_colors, color_hist, brightness, texture_hist, edge_density, edge_orient_hist, style_vector
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Unsupervised ML: K-Means Color Palette
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
    if len(pixels) > 5000:
        pixels = pixels[np.random.choice(len(pixels), 5000, replace=False)]
    
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.tolist()
    
    # 2. Color Distribution: HSV Histogram (8x8 H-S bins)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten().tolist()
    
    # 3. Brightness & Contrast & Saturation
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 128.0
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0
    
    # 4. Texture Descriptor: Local Binary Pattern (LBP)
    gray_small = cv2.resize(gray, (128, 128))
    lbp = local_binary_pattern(gray_small, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = (lbp_hist / (lbp_hist.sum() + 1e-7)).tolist()
    
    # 5. Shape/Edge Features: Canny Edge Density & Orientation Histogram
    edges = cv2.Canny(gray_small, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    
    gx = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    edge_pixels = edges > 0
    if np.any(edge_pixels):
        edge_orient_hist, _ = np.histogram(angle[edge_pixels], bins=8, range=(0, 360))
        edge_orient_hist = (edge_orient_hist / (edge_orient_hist.sum() + 1e-7)).tolist()
    else:
        edge_orient_hist = [0.125] * 8
        
    # 6. Measurable Visual Style Feature Vector [brightness, contrast, saturation, lbp_std]
    style_vector = [brightness, contrast, saturation, float(np.std(lbp_hist))]
    
    return {
        "dominant_colors": dominant_colors,
        "color_hist": hsv_hist,
        "brightness": brightness,
        "texture_hist": lbp_hist,
        "edge_density": edge_density,
        "edge_orient_hist": edge_orient_hist,
        "style_vector": style_vector
    }

def build_features():
    print("=== VisionHome ML Feature Precomputation Pipeline ===")
    
    existing_data = {}
    if METADATA_FILE.exists():
        print(f"Loading existing metadata from {METADATA_FILE}...")
        try:
            with open(METADATA_FILE, "rb") as f:
                existing_data = pickle.load(f)
        except Exception as e:
            print(f"Failed to load existing metadata: {e}")
            existing_data = {}
            
    resnet_model = None
    if TF_AVAILABLE:
        print("Initializing pretrained ResNet50 model for visual feature extraction...")
        resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
    dataset_metadata = {}
    start_time = time.time()
    processed_count = 0
    
    all_image_paths = []
    for root, dirs, files in os.walk(DATASETS_DIR):
        for f in files:
            if Path(f).suffix.lower() in ALLOWED_EXT:
                abs_path = Path(root) / f
                rel_path = str(abs_path.relative_to(DATASETS_DIR))
                all_image_paths.append((rel_path, abs_path))
                
    total_images = len(all_image_paths)
    print(f"Found {total_images} dataset images.")
    
    for idx, (rel_path, abs_path) in enumerate(all_image_paths, 1):
        if idx % 500 == 0 or idx == total_images:
            print(f"[{idx}/{total_images}] Precomputing ML features...")
            
        img_bgr = cv2.imread(str(abs_path))
        if img_bgr is None:
            continue
            
        db_features = None
        if rel_path in existing_data and "features" in existing_data[rel_path]:
            db_features = existing_data[rel_path]["features"]
            
        if db_features is None and resnet_model is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            x = keras_image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            raw_features = resnet_model.predict(x, verbose=0)[0]
            db_features = raw_features / (np.linalg.norm(raw_features) + 1e-10)
            
        cv_feats = None
        if (rel_path in existing_data and 
            "color_hist" in existing_data[rel_path] and 
            "texture_hist" in existing_data[rel_path]):
            cv_feats = {
                "dominant_colors": existing_data[rel_path].get("dominant_colors", []),
                "color_hist": existing_data[rel_path].get("color_hist", []),
                "brightness": existing_data[rel_path].get("brightness", 0.5),
                "texture_hist": existing_data[rel_path].get("texture_hist", []),
                "edge_density": existing_data[rel_path].get("edge_density", 0.05),
                "edge_orient_hist": existing_data[rel_path].get("edge_orient_hist", []),
                "style_vector": existing_data[rel_path].get("style_vector", [0.5, 0.2, 0.2, 0.05])
            }
        else:
            cv_feats = extract_cv_features(img_bgr)
            
        dataset_metadata[rel_path] = {
            "features": db_features,
            "dominant_colors": cv_feats["dominant_colors"],
            "color_hist": cv_feats["color_hist"],
            "brightness": cv_feats["brightness"],
            "texture_hist": cv_feats["texture_hist"],
            "edge_density": cv_feats["edge_density"],
            "edge_orient_hist": cv_feats["edge_orient_hist"],
            "style_vector": cv_feats["style_vector"],
            "style": existing_data.get(rel_path, {}).get("style", "modern")
        }
        processed_count += 1
        
    print(f"Saving precomputed ML dataset metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(dataset_metadata, f)
        
    elapsed = time.time() - start_time
    print(f"Done! Successfully precomputed ML features for {processed_count} images in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    build_features()
