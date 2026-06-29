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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

try:
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ================================
# Paths
# ================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
TMP_DIR = PROJECT_ROOT / "flask_backend_tmp"

# Allowed extensions
ALLOWED_EXT = [".jpg", ".jpeg", ".png"]

# Create temp folder
os.makedirs(TMP_DIR, exist_ok=True)

# ================================
# Flask App
# ================================
app = Flask(__name__)
CORS(app)

METADATA_FILE = PROJECT_ROOT / "dataset_metadata.pkl"
dataset_metadata = {}
if METADATA_FILE.exists():
    with open(METADATA_FILE, "rb") as f:
        dataset_metadata = pickle.load(f)

resnet_model = None
if TF_AVAILABLE:
    print("Loading ResNet50 for Perception Layer...")
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_image_features(img_bytes):
    if not TF_AVAILABLE or resnet_model is None:
        return None, [], "modern"
        
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img_cv is None:
        return None, [], "modern"
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Context Awareness (Style & Color)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    pixels = img_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    if len(pixels) > 10000:
        pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]
        
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.tolist()
    
    if brightness > 180 or (brightness > 150 and contrast < 50):
        style = "minimal"
    elif brightness < 100 or contrast > 75:
        style = "luxury"
    else:
        style = "modern"
        
    img_resized = cv2.resize(img_rgb, (224, 224))
    x = keras_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = resnet_model.predict(x, verbose=0)[0]
    features_norm = features / (np.linalg.norm(features) + 1e-10)
    
    return features_norm, colors, style

# ================================
# Helper Functions
# ================================
def sanitize_token(tok):
    """Lowercase and remove unwanted characters"""
    return re.sub(r'[^a-z0-9]', '', tok.lower())

def filename_to_tags(fname):
    """Generate tags from filename"""
    base = Path(fname).name
    name, _ = os.path.splitext(base)

    s = re.sub(r'[-\s]+', '-', name)
    parts = re.split(r'[_|\W]+', s)

    tags = []
    for p in parts:
        tok = sanitize_token(p)

        if not tok or tok.isdigit() or len(tok) < 2:
            continue

        tags.append(tok)

    return list(dict.fromkeys(tags))


FURNITURE_CATEGORIES = [
    "sofa",
    "chair",
    "bed",
    "table",
    "swivelchair"
]

# ================================
# Build Dataset Index
# ================================
def build_index():

    index = {}

    for root, dirs, files in os.walk(DATASETS_DIR):

        for f in files:

            if Path(f).suffix.lower() not in ALLOWED_EXT:
                continue

            abs_path = Path(root) / f
            rel_path = abs_path.relative_to(DATASETS_DIR)

            parts = rel_path.parts

            # category usually second last folder
            if len(parts) >= 2:
                category = parts[-2].lower()
            else:
                category = "unknown"

            index[str(rel_path)] = [category]

    return index


TAGS_INDEX = build_index()

# ================================
# Routes
# ================================

@app.route("/")
def home():
    """Root route"""
    return jsonify({
        "message": "VisionHome Backend Running",
        "endpoints": [
            "/health",
            "/tags",
            "/recommend",
            "/image/<filename>",
            "/prepare"
        ]
    })


@app.route("/health")
def health():
    """Server health check"""
    return jsonify({
        "status": "ok",
        "count": len(TAGS_INDEX),
        "categories": FURNITURE_CATEGORIES
    })


@app.route("/tags")
def tags():
    """Return dataset index"""
    return jsonify(TAGS_INDEX)


@app.route("/recommend", methods=["POST"])
def recommend():
    """Recommend furniture images based on query"""

    data = request.get_json(force=True) or {}
    q = (data.get("q") or "").strip().lower()

    if not q:
        return jsonify({"results": []})

    # Basic NLP parsing for synonyms
    SYNONYMS = {
        "couch": "sofa",
        "seat": "chair",
        "stool": "chair",
        "desk": "table",
        "nightstand": "table",
        "mattress": "bed"
    }

    raw_tokens = [sanitize_token(p) for p in q.split()]
    query_tokens = [SYNONYMS.get(t, t) for t in raw_tokens]

    # Style aware parsing
    style_filters = [t for t in query_tokens if t in ["modern", "minimal", "luxury"]]
    query_tokens = [t for t in query_tokens if t not in ["modern", "minimal", "luxury"]]

    matched_categories = [
        cat for cat in FURNITURE_CATEGORIES
        if cat in query_tokens
    ]

    results = []

    # Category based search
    if matched_categories:
        for rel, tags in TAGS_INDEX.items():
            if any(cat in tags for cat in matched_categories):
                results.append(rel)
    else:
        # tag based search
        for rel, tags in TAGS_INDEX.items():
            if any(token in tags for token in query_tokens):
                results.append(rel)

    # fallback partial match
    if not results:
        for rel, tags in TAGS_INDEX.items():
            if any(any(token in t for t in tags) for token in query_tokens):
                results.append(rel)

    # Apply style filter from NLP
    if style_filters:
        filtered_results = []
        for rel in results:
            meta = dataset_metadata.get(rel, {})
            db_style = meta.get("style", "modern")
            if any(s == db_style for s in style_filters):
                filtered_results.append(rel)
        if filtered_results:
            results = filtered_results

    random.shuffle(results)

    final_results = [{
        "item": r,
        "explanation": "Keyword match based on your search terms.",
        "badge": "Search",
        "sim_visual": 1.0, "sim_color": 1.0, "sim_style": 1.0, "final_score": 1.0
    } for r in results[:16]]

    return jsonify({
        "message": f"Search results for '{q}'",
        "groups": {"🔍 Search Results": final_results}
    })

def color_distance(c1, c2):
    # Euclidean distance between RGB vectors
    # c1 is list of 3 colors, we will compute min distance between any of the 3
    min_dist = float('inf')
    for co1 in c1:
        for co2 in c2:
            d = np.linalg.norm(np.array(co1) - np.array(co2))
            if d < min_dist:
                min_dist = d
    return min_dist


@app.route("/ai-recommend", methods=["POST"])
def ai_recommend():
    """AI Recommendation based on image features, style, and color"""
    if 'room_image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    color_weight_param = float(request.form.get("color_weight", 0.5))
    style_weight_param = 1.0 - color_weight_param
        
    file = request.files['room_image']
    img_bytes = file.read()
    
    room_features, room_colors, room_style = extract_image_features(img_bytes)
    
    if room_features is None:
        return jsonify({"groups": {}})
        
    scores = []
    
    for rel_path, meta in dataset_metadata.items():
        db_features = meta.get("features")
        db_colors = meta.get("dominant_colors", [])
        db_style = meta.get("style", "modern")
        
        category = TAGS_INDEX.get(rel_path, ["unknown"])[0]
        
        if db_features is None:
            continue
            
        sim_visual = float(np.dot(room_features, db_features))
        dist_color = color_distance(room_colors, db_colors)
        sim_color = max(0, 1.0 - (dist_color / 255.0))
        sim_style = 1.0 if room_style == db_style else 0.5
        
        # Weighted reasoning based on user parameter
        final_score = (0.5 * sim_visual) + 0.5 * ((color_weight_param * sim_color) + (style_weight_param * sim_style))
        
        # Generate XAI Explanation
        explanations = []
        badge = ""
        highest_sub = max(sim_visual, sim_color, sim_style)
        
        if highest_sub == sim_visual and sim_visual > 0.8:
            explanations.append("Highly visually similar using AI feature extraction")
            badge = "Best Match"
        elif highest_sub == sim_color and sim_color > 0.8:
            explanations.append("Matches dominant color palette")
            badge = "Color Match"
        elif highest_sub == sim_style and sim_style == 1.0:
            explanations.append(f"Fits detected {db_style} style")
            badge = "Style Match"
        else:
            explanations.append("Balanced match across style, color, and visual similarity")
            badge = "AI Pick"
            
        scores.append({
            "item": rel_path,
            "category": category,
            "explanation": explanations[0],
            "badge": badge,
            "sim_visual": sim_visual,
            "sim_color": sim_color,
            "sim_style": sim_style,
            "final_score": final_score
        })
        
    scores.sort(key=lambda x: x["final_score"], reverse=True)
    
    seen_paths = set()
    
    def get_top_unique(item_list, limit, filter_fn=None, sort_key="final_score"):
        sorted_list = sorted(item_list, key=lambda x: x.get(sort_key, 0), reverse=True)
        results = []
        for v in sorted_list:
            if v["item"] not in seen_paths and (filter_fn is None or filter_fn(v)):
                results.append(v)
                seen_paths.add(v["item"])
                if len(results) >= limit:
                    break
        return results

    groups = {}
    groups["🔥 Top AI Picks"] = get_top_unique(scores, 4)
    groups["🎨 Color Matches"] = get_top_unique(scores, 4, sort_key="sim_color")
    groups["🪑 Chairs"] = get_top_unique(scores, 4, lambda x: x["category"] in ["chair", "swivelchair"])
    groups["🛋 Sofas"] = get_top_unique(scores, 4, lambda x: x["category"] == "sofa")
    groups["🛏 Beds"] = get_top_unique(scores, 4, lambda x: x["category"] == "bed")

    # Personalized Header Context
    tone_str = "neutral"
    if room_colors:
        avg_col = sum(room_colors[0]) / 3
        if avg_col > 180: tone_str = "light"
        elif avg_col < 80: tone_str = "dark"
        else: tone_str = "warm"

    message = f"We detected a {room_style}-style room with {tone_str} tones. Here are AI-selected furniture recommendations."

    return jsonify({
        "message": message,
        "groups": groups
    })


@app.route("/image/<path:filename>")
def serve_image(filename):
    """Serve dataset images"""

    requested = Path(filename)
    abs_path = DATASETS_DIR / requested

    try:
        abs_path.relative_to(DATASETS_DIR)
    except Exception:
        abort(404)

    if not abs_path.exists() or not abs_path.is_file():
        abort(404)

    return send_file(str(abs_path), conditional=True)


@app.route("/prepare", methods=["POST"])
def prepare():
    """
    Remove background from furniture image
    """

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

    # Read file
    with open(abs_path, "rb") as f:
        inp = f.read()

    # Remove background
    try:
        out_bytes = remove(inp)

    except Exception as e:
        return jsonify({
            "error": f"background removal failed: {str(e)}"
        }), 500

    return send_file(
        io.BytesIO(out_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name=f"processed_{requested.name}"
    )


# ================================
# Run Server
# ================================
if __name__ == "__main__":

    print("Starting VisionHome Flask Backend...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Datasets: {DATASETS_DIR}")
    print(f"Categories: {FURNITURE_CATEGORIES}")
    print(f"Indexed items: {len(TAGS_INDEX)}")

    app.run(
        host="0.0.0.0",
        port=5050,
        debug=True
    )