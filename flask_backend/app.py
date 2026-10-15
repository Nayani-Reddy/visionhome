"""
VisionHome Flask Backend Server
===============================
This server provides API endpoints for:
1. Mode 1: ML Recommendations (Search Query Candidate Filtering + ML Ranking) (/recommend)
2. Mode 2: AI Recommendations (Room Image Feature Extraction + Automatic ML Ranking) (/ai-recommend)
3. Background Removal Pipeline using rembg / U2-Net (/prepare)
4. Static Image & Dataset Asset Serving (/image/<filename>)

EXPLICIT ARCHITECTURAL CLASSIFICATION OF COMPONENTS:
----------------------------------------------------
1. DEEP LEARNING (DL):
   - Pretrained ResNet50 Convolutional Neural Network (ImageNet weights, top layer excluded, global average pooling).
   - Loaded ONLY ONCE at Flask server startup.
   - Extracts 2048-dimensional visual feature vectors for room and dataset furniture images.

2. UNSUPERVISED MACHINE LEARNING (ML):
   - K-Means Clustering (k=3 centroids) on RGB image pixel samples.
   - Extracts dominant color palette centroids to compute color palette compatibility.

3. COMPUTER VISION & IMAGE PROCESSING FEATURE EXTRACTION:
   - Color Distribution: Normalized 2D HSV color histogram (8x8 bins for H & S).
   - Brightness Match: Normalized mean grayscale intensity (0.0 to 1.0).
   - Micro-Texture Descriptor: Local Binary Pattern (LBP, P=8, R=1, 10-bin uniform histogram).
   - Shape/Edge Structure: Canny edge detector density & 8-bin edge orientation histogram.
   - Visual Style Feature Vector: Measurable vector composed of brightness, contrast, saturation, and texture variance.

4. CONVENTIONAL APPLICATION LOGIC & METADATA:
   - Dataset directory category labels (sofa, chair, bed, table, swivelchair).
   - Text parsing, candidate filtering, background removal via rembg, and asset serving.
"""

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import os
import re
import io
import random
import pickle
import numpy as np
import cv2
from pathlib import Path
from rembg import remove
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern

try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ================================
# Paths & Configuration
# ================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
TMP_DIR = PROJECT_ROOT / "flask_backend_tmp"
ALLOWED_EXT = [".jpg", ".jpeg", ".png"]

os.makedirs(TMP_DIR, exist_ok=True)

# ==============================================================================
# MACHINE LEARNING RECOMMENDATION SYSTEM WEIGHT CONFIGURATION
# Centralized Configuration Section for Weighted Multi-Parameter ML Scoring
# Total Weight Sum = 1.00 (100%)
# ==============================================================================
ML_WEIGHTS = {
    "visual_similarity": 0.40,       # 40%: Deep Learning ResNet50 Feature Vector Cosine Similarity
    "color_compatibility": 0.15,     # 15%: Unsupervised ML K-Means Dominant Color Palette Distance
    "color_distribution": 0.10,      # 10%: HSV Color Histogram Intersection
    "brightness_compatibility": 0.10, # 10%: Grayscale Mean Brightness Match
    "texture_similarity": 0.10,      # 10%: Local Binary Pattern (LBP) Texture Match
    "shape_similarity": 0.05,        #  5%: Canny Edge Density & Orientation Match
    "category_relevance": 0.05,      #  5%: Dataset Category Metadata Relevance
    "style_feature": 0.05            #  5%: Visual-Style Heuristic Feature Vector Match
}

# Supported Furniture Categories
FURNITURE_CATEGORIES = ["sofa", "chair", "bed", "table", "swivelchair"]

# Global store for latest uploaded room features
latest_room_feats = None

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__)
CORS(app)

# Load Precomputed Dataset Metadata
METADATA_FILE = PROJECT_ROOT / "dataset_metadata.pkl"
dataset_metadata = {}
if METADATA_FILE.exists():
    with open(METADATA_FILE, "rb") as f:
        dataset_metadata = pickle.load(f)
    print(f"Loaded {len(dataset_metadata)} precomputed ML dataset records.")

# Load Pretrained ResNet50 Model ONLY ONCE at Startup
resnet_model = None
if TF_AVAILABLE:
    print("Loading Pretrained ResNet50 CNN model for visual feature extraction...")
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Build Tag & Category Index for Dataset
def build_index():
    index = {}
    for root, dirs, files in os.walk(DATASETS_DIR):
        for f in files:
            if Path(f).suffix.lower() not in ALLOWED_EXT:
                continue
            abs_path = Path(root) / f
            rel_path = abs_path.relative_to(DATASETS_DIR)
            parts = rel_path.parts
            if len(parts) >= 2:
                category = parts[-2].lower()
            else:
                category = "unknown"
            index[str(rel_path)] = [category]
    return index

TAGS_INDEX = build_index()

# ==============================================================================
# FEATURE EXTRACTION PIPELINE FOR UPLOADED ROOM IMAGES
# ==============================================================================
def extract_room_ml_features(img_bytes):
    """
    Extracts all 8 feature representations from an uploaded room image bytes.
    - ResNet50 Deep Feature Vector (2048-dim, normalized)
    - K-Means Dominant Colors (3 cluster centroids)
    - HSV Color Histogram (8x8 H-S bins)
    - Grayscale Brightness & Contrast
    - LBP Texture Descriptor Histogram (10 bins)
    - Canny Edge Density & Orientation Histogram (8 bins)
    - Visual Style Feature Vector [brightness, contrast, saturation, lbp_std]
    """
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Deep Learning Feature Extraction via Pretrained ResNet50
    resnet_features = None
    if resnet_model is not None:
        img_resized = cv2.resize(img_rgb, (224, 224))
        x = keras_image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        raw_feat = resnet_model.predict(x, verbose=0)[0]
        resnet_features = raw_feat / (np.linalg.norm(raw_feat) + 1e-10)

    # 2. Unsupervised ML: K-Means Clustering for Dominant Color Palette
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)
    if len(pixels) > 5000:
        pixels = pixels[np.random.choice(len(pixels), 5000, replace=False)]
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=42)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.tolist()

    # 3. Image Processing: HSV Color Distribution Histogram
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten().tolist()

    # 4. Image Processing: Brightness, Contrast & Saturation
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 128.0
    saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    # 5. Computer Vision: Local Binary Pattern (LBP) Micro-Texture Descriptor
    gray_small = cv2.resize(gray, (128, 128))
    lbp = local_binary_pattern(gray_small, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = (lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-7)).tolist()

    # 6. Computer Vision: Canny Edge Density & Edge Orientation
    edges = cv2.Canny(gray_small, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    gx = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    edge_pixels = edges > 0
    if np.any(edge_pixels):
        edge_orient_hist, _ = np.histogram(angle[edge_pixels], bins=8, range=(0, 360))
        edge_orient_hist = (edge_orient_hist.astype(np.float32) / (edge_orient_hist.sum() + 1e-7)).tolist()
    else:
        edge_orient_hist = [0.125] * 8

    # 7. Visual Style Feature Vector (Measurable Low-Level Image Characteristics)
    style_vector = [brightness, contrast, saturation, float(np.std(lbp_hist))]

    return {
        "features": resnet_features,
        "dominant_colors": dominant_colors,
        "color_hist": hsv_hist,
        "brightness": brightness,
        "texture_hist": lbp_hist,
        "edge_density": edge_density,
        "edge_orient_hist": edge_orient_hist,
        "style_vector": style_vector
    }

# ==============================================================================
# MULTI-PARAMETER ML SCORE CALCULATOR
# ==============================================================================
def calculate_ml_recommendation_score(room_feats, db_meta, category):
    """
    Calculates 8 normalized parameter scores (0.0 to 1.0) and computes weighted final score:
    1. Visual Feature Similarity (40%)
    2. Color Compatibility (15%)
    3. Color Distribution Similarity (10%)
    4. Brightness Compatibility (10%)
    5. Texture Similarity (10%)
    6. Shape/Edge Similarity (5%)
    7. Furniture Category Relevance (5%)
    8. Visual Style Feature (5%)
    """
    # 1. Visual Feature Similarity (ResNet50 Cosine Similarity)
    db_features = db_meta.get("features")
    if db_features is not None and room_feats.get("features") is not None and np.any(room_feats["features"]):
        sim_visual = float(np.dot(room_feats["features"], db_features))
        sim_visual = max(0.0, min(1.0, sim_visual))
    else:
        # If no room vector uploaded yet, use feature magnitude as reference visual score
        if db_features is not None:
            sim_visual = float(np.linalg.norm(db_features) / (np.linalg.norm(db_features) + 1.0))
            sim_visual = max(0.2, min(0.95, sim_visual))
        else:
            sim_visual = 0.50

    # 2. Color Compatibility (K-Means Dominant Color Palette Distance)
    c_room = np.array(room_feats.get("dominant_colors", []), dtype=np.float32)
    c_furn = np.array(db_meta.get("dominant_colors", []), dtype=np.float32)
    if len(c_room) > 0 and len(c_furn) > 0:
        dists = [np.min(np.linalg.norm(c_room - cf, axis=1)) for cf in c_furn]
        mean_dist = float(np.mean(dists))
        sim_color = max(0.0, min(1.0, 1.0 - (mean_dist / 255.0)))
    else:
        sim_color = 0.50

    # 3. Color Distribution Similarity (HSV Histogram Intersection)
    h_room = np.array(room_feats.get("color_hist", []), dtype=np.float32)
    h_furn = np.array(db_meta.get("color_hist", []), dtype=np.float32)
    if len(h_room) > 0 and len(h_furn) > 0 and len(h_room) == len(h_furn):
        sim_color_dist = float(np.sum(np.minimum(h_room, h_furn)))
        sim_color_dist = max(0.0, min(1.0, sim_color_dist))
    else:
        sim_color_dist = 0.50

    # 4. Brightness Compatibility (Grayscale Brightness Alignment)
    b_room = float(room_feats.get("brightness", 0.5))
    b_furn = float(db_meta.get("brightness", 0.5))
    sim_brightness = max(0.0, min(1.0, 1.0 - abs(b_room - b_furn)))

    # 5. Texture Similarity (LBP Descriptor Histogram Intersection)
    t_room = np.array(room_feats.get("texture_hist", []), dtype=np.float32)
    t_furn = np.array(db_meta.get("texture_hist", []), dtype=np.float32)
    if len(t_room) > 0 and len(t_furn) > 0 and len(t_room) == len(t_furn):
        sim_texture = float(np.sum(np.minimum(t_room, t_furn)))
        sim_texture = max(0.0, min(1.0, sim_texture))
    else:
        sim_texture = 0.50

    # 6. Shape/Edge Similarity (Canny Edge Density & Edge Orientation Match)
    e_room = float(room_feats.get("edge_density", 0.05))
    e_furn = float(db_meta.get("edge_density", 0.05))
    density_match = max(0.0, min(1.0, 1.0 - abs(e_room - e_furn) * 5.0))

    o_room = np.array(room_feats.get("edge_orient_hist", []), dtype=np.float32)
    o_furn = np.array(db_meta.get("edge_orient_hist", []), dtype=np.float32)
    if len(o_room) > 0 and len(o_furn) > 0 and len(o_room) == len(o_furn):
        orient_match = float(np.sum(np.minimum(o_room, o_furn)))
    else:
        orient_match = 0.50
    sim_shape = max(0.0, min(1.0, 0.5 * density_match + 0.5 * orient_match))

    # 7. Furniture Category Relevance (Dataset Metadata Category Validation)
    category_relevance = 1.00 if category in FURNITURE_CATEGORIES else 0.70

    # 8. Visual Style Feature (Measurable Low-Level Attribute Vector Distance)
    s_room = np.array(room_feats.get("style_vector", [0.5, 0.2, 0.2, 0.05]), dtype=np.float32)
    s_furn = np.array(db_meta.get("style_vector", [0.5, 0.2, 0.2, 0.05]), dtype=np.float32)
    style_dist = float(np.linalg.norm(s_room - s_furn))
    sim_style = max(0.0, min(1.0, 1.0 - (style_dist / 2.0)))

    # Compute Final ML Recommendation Score using Centralized Configuration Weights
    final_score = (
        ML_WEIGHTS["visual_similarity"] * sim_visual +
        ML_WEIGHTS["color_compatibility"] * sim_color +
        ML_WEIGHTS["color_distribution"] * sim_color_dist +
        ML_WEIGHTS["brightness_compatibility"] * sim_brightness +
        ML_WEIGHTS["texture_similarity"] * sim_texture +
        ML_WEIGHTS["shape_similarity"] * sim_shape +
        ML_WEIGHTS["category_relevance"] * category_relevance +
        ML_WEIGHTS["style_feature"] * sim_style
    )

    return {
        "visual_similarity": round(sim_visual, 2),
        "color_compatibility": round(sim_color, 2),
        "color_distribution": round(sim_color_dist, 2),
        "brightness_match": round(sim_brightness, 2),
        "texture_similarity": round(sim_texture, 2),
        "shape_similarity": round(sim_shape, 2),
        "category_relevance": round(category_relevance, 2),
        "style_feature": round(sim_style, 2),
        "final_score": round(float(final_score), 2)
    }

# Helper functions for text tags
def sanitize_token(tok):
    return re.sub(r'[^a-z0-9]', '', tok.lower())

# ================================
# Flask Web API Routes
# ================================

@app.route("/")
def home():
    return jsonify({
        "message": "VisionHome Machine Learning Backend Running",
        "ml_weights": ML_WEIGHTS,
        "endpoints": [
            "/health",
            "/tags",
            "/recommend",
            "/ai-recommend",
            "/image/<filename>",
            "/prepare"
        ]
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset_count": len(TAGS_INDEX),
        "categories": FURNITURE_CATEGORIES,
        "ml_model": "Pretrained ResNet50 Feature Extractor & Multi-Parameter Weighted Evaluator"
    })

@app.route("/tags")
def tags():
    return jsonify(TAGS_INDEX)

# MODE 1: ML Recommendations (Search Query Candidate Filtering + ML Ranking)
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Mode 1: ML Recommendations based on Search Query
    Filters candidate furniture items by search query/category,
    then ranks candidates using the multi-parameter ML recommendation system.
    """
    global latest_room_feats

    data = request.get_json(force=True) or {}
    q = (data.get("q") or "").strip().lower()

    if not q:
        return jsonify({"results": []})

    SYNONYMS = {
        "couch": "sofa", "seat": "chair", "stool": "chair",
        "desk": "table", "nightstand": "table", "mattress": "bed"
    }

    raw_tokens = [sanitize_token(p) for p in q.split()]
    query_tokens = [SYNONYMS.get(t, t) for t in raw_tokens]

    matched_categories = [cat for cat in FURNITURE_CATEGORIES if cat in query_tokens]
    matched_paths = []

    if matched_categories:
        for rel, tags_list in TAGS_INDEX.items():
            if any(cat in tags_list for cat in matched_categories):
                matched_paths.append(rel)
    else:
        for rel, tags_list in TAGS_INDEX.items():
            if any(token in tags_list for token in query_tokens):
                matched_paths.append(rel)

    if not matched_paths:
        for rel, tags_list in TAGS_INDEX.items():
            if any(any(token in t for t in tags_list) for token in query_tokens):
                matched_paths.append(rel)

    # Use latest room features if available, otherwise construct baseline room features
    ref_room_feats = latest_room_feats
    if ref_room_feats is None:
        ref_room_feats = {
            "features": np.zeros(2048, dtype=np.float32),
            "dominant_colors": [[180, 170, 160], [100, 90, 80], [220, 210, 200]],
            "color_hist": [0.015] * 64,
            "brightness": 0.55,
            "texture_hist": [0.10] * 10,
            "edge_density": 0.05,
            "edge_orient_hist": [0.125] * 8,
            "style_vector": [0.55, 0.25, 0.25, 0.05]
        }

    scored_results = []
    for rel_path in matched_paths:
        db_meta = dataset_metadata.get(rel_path, {})
        category = TAGS_INDEX.get(rel_path, ["unknown"])[0]
        
        score_dict = calculate_ml_recommendation_score(ref_room_feats, db_meta, category)
        
        if matched_categories and category in matched_categories:
            score_dict["category_relevance"] = 1.00

        # Recalculate final score
        final_score = (
            ML_WEIGHTS["visual_similarity"] * score_dict["visual_similarity"] +
            ML_WEIGHTS["color_compatibility"] * score_dict["color_compatibility"] +
            ML_WEIGHTS["color_distribution"] * score_dict["color_distribution"] +
            ML_WEIGHTS["brightness_compatibility"] * score_dict["brightness_match"] +
            ML_WEIGHTS["texture_similarity"] * score_dict["texture_similarity"] +
            ML_WEIGHTS["shape_similarity"] * score_dict["shape_similarity"] +
            ML_WEIGHTS["category_relevance"] * score_dict["category_relevance"] +
            ML_WEIGHTS["style_feature"] * score_dict["style_feature"]
        )
        score_dict["final_score"] = round(float(final_score), 2)
        
        scored_results.append({
            "item": rel_path,
            "category": category,
            "badge": "ML Picked",
            "explanation": f"Filtered by query '{q}' & ranked by Weighted ML Score ({score_dict['final_score']:.2f}).",
            "visual_similarity": score_dict["visual_similarity"],
            "color_compatibility": score_dict["color_compatibility"],
            "color_distribution": score_dict["color_distribution"],
            "brightness_match": score_dict["brightness_match"],
            "texture_similarity": score_dict["texture_similarity"],
            "shape_similarity": score_dict["shape_similarity"],
            "category_relevance": score_dict["category_relevance"],
            "style_feature": score_dict["style_feature"],
            "final_score": score_dict["final_score"]
        })

    # Rank search candidates using Weighted ML Score
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    return jsonify({
        "message": f"ML-Ranked search candidates for '{q}'",
        "groups": {f"🔍 ML Ranked Results for '{q}'": scored_results[:16]}
    })

# MODE 2: AI Recommendations (Room Image Upload without Search)
@app.route("/ai-recommend", methods=["POST"])
def ai_recommend():
    """
    Mode 2: AI Recommendations based on Uploaded Room Image
    Extracts room image features, evaluates 8 ML parameters against dataset items,
    computes weighted ML scores, sorts recommendations, and enforces category diversity.
    """
    global latest_room_feats

    if 'room_image' not in request.files:
        return jsonify({"error": "No room image uploaded"}), 400

    file = request.files['room_image']
    img_bytes = file.read()

    room_feats = extract_room_ml_features(img_bytes)
    if room_feats is None:
        return jsonify({"error": "Failed to decode uploaded room image"}), 400

    # Store latest room features for search query ranking context
    latest_room_feats = room_feats

    all_scores = []

    for rel_path, db_meta in dataset_metadata.items():
        category = TAGS_INDEX.get(rel_path, ["unknown"])[0]
        
        score_dict = calculate_ml_recommendation_score(room_feats, db_meta, category)
        
        highest_param = max(
            [("Visual Similarity", score_dict["visual_similarity"]),
             ("Color Match", score_dict["color_compatibility"]),
             ("Texture Fit", score_dict["texture_similarity"]),
             ("Brightness Fit", score_dict["brightness_match"])],
            key=lambda x: x[1]
        )
        
        if score_dict["final_score"] >= 0.80:
            badge = "Top AI Pick"
            explanation = f"High overall AI recommendation match ({score_dict['final_score']:.2f})."
        elif highest_param[0] == "Color Match":
            badge = "Color Match"
            explanation = f"Matches dominant room color palette ({score_dict['color_compatibility']:.2f})."
        elif highest_param[0] == "Visual Similarity":
            badge = "Visual Match"
            explanation = f"Strong visual feature vector similarity ({score_dict['visual_similarity']:.2f})."
        else:
            badge = "AI Pick"
            explanation = f"Balanced match across multi-parameter AI evaluation ({score_dict['final_score']:.2f})."

        item_result = {
            "item": rel_path,
            "category": category,
            "badge": badge,
            "explanation": explanation,
            "visual_similarity": score_dict["visual_similarity"],
            "color_compatibility": score_dict["color_compatibility"],
            "color_distribution": score_dict["color_distribution"],
            "brightness_match": score_dict["brightness_match"],
            "texture_similarity": score_dict["texture_similarity"],
            "shape_similarity": score_dict["shape_similarity"],
            "category_relevance": score_dict["category_relevance"],
            "style_feature": score_dict["style_feature"],
            "final_score": score_dict["final_score"]
        }
        all_scores.append(item_result)

    # Sort all furniture by final weighted ML recommendation score
    all_scores.sort(key=lambda x: x["final_score"], reverse=True)

    # Category Diversity Selection
    seen_paths = set()
    def get_diverse_top(items, limit, filter_fn=None, sort_key="final_score"):
        sorted_items = sorted(items, key=lambda x: x.get(sort_key, 0), reverse=True)
        results = []
        for v in sorted_items:
            if v["item"] not in seen_paths and (filter_fn is None or filter_fn(v)):
                results.append(v)
                seen_paths.add(v["item"])
                if len(results) >= limit:
                    break
        return results

    groups = {}
    groups["🌟 Top ML Recommendations"] = get_diverse_top(all_scores, 4)
    groups["🎨 Color & Palette Matches"] = get_diverse_top(all_scores, 4, sort_key="color_compatibility")
    groups["🪑 Chairs"] = get_diverse_top(all_scores, 4, lambda x: x["category"] in ["chair", "swivelchair"])
    groups["🛋 Sofas"] = get_diverse_top(all_scores, 4, lambda x: x["category"] == "sofa")
    groups["🛏 Beds"] = get_diverse_top(all_scores, 4, lambda x: x["category"] == "bed")
    groups["🪵 Tables"] = get_diverse_top(all_scores, 4, lambda x: x["category"] == "table")

    message = f"AI Recommendations generated from uploaded room image using ResNet50 deep features, K-Means color clustering, and weighted ML evaluation."

    return jsonify({
        "message": message,
        "weights_used": ML_WEIGHTS,
        "groups": groups
    })

# Serve Dataset Image Assets
@app.route("/image/<path:filename>")
def serve_image(filename):
    requested = Path(filename)
    abs_path = DATASETS_DIR / requested
    try:
        abs_path.relative_to(DATASETS_DIR)
    except Exception:
        abort(404)
    if not abs_path.exists() or not abs_path.is_file():
        abort(404)
    return send_file(str(abs_path), conditional=True)

# Background Removal Tool Endpoint (Visualization / Application Feature)
@app.route("/prepare", methods=["POST"])
def prepare():
    data = request.get_json(force=True) or {}
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "filename missing"}), 400

    requested = Path(filename)
    abs_path = DATASETS_DIR / requested

    try:
        abs_path.relative_to(DATASETS_DIR)
    except Exception:
        return jsonify({"error": "not found"}), 404

    if not abs_path.exists() or not abs_path.is_file():
        return jsonify({"error": "not found"}), 404

    with open(abs_path, "rb") as f:
        inp = f.read()

    try:
        out_bytes = remove(inp)
    except Exception as e:
        return jsonify({"error": f"background removal failed: {str(e)}"}), 500

    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"processed_{requested.name}"
    )

if __name__ == "__main__":
    print("Starting VisionHome Flask ML Server...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Loaded precomputed metadata records: {len(dataset_metadata)}")
    
    app.run(
        host="0.0.0.0",
        port=5050,
        debug=True
    )