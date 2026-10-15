# VisionHome – Intelligent Interior Design & Furniture Recommendation System

VisionHome is a state-of-the-art, machine learning-driven web application that allows users to seamlessly design interior spaces. Users can upload a photo of an empty room, receive automated AI-driven furniture recommendations, dynamically place objects in the room (with automatic background removal), and tweak contextual parameters like color, texture, and rotation. 

The core feature of VisionHome is its highly sophisticated, 8-parameter **Machine Learning Recommendation Engine** backed by Explainable AI (XAI) principles.

---

## 🏗 System Architecture Overview

The system is separated into a lightweight frontend (Vanilla HTML/CSS/JS) and a robust Python-based backend (Flask) handling heavy Machine Learning and Computer Vision tasks.

### 1. Frontend (User Interface)
- **Stack:** Vanilla HTML5, CSS3, JavaScript.
- **Design Philosophy:** A spacious, responsive layout prioritizing the visual aspects of furniture discovery. The application uses a dynamic grid system (3-4 items on desktop) preventing duplicate items.
- **Features:**
  - **Interactive Canvas Editor:** Move, rotate, scale, and manipulate furniture layers. Contextual editing panels to modify colors, saturation, and apply texture overlays (Velvet, Leather, Boucle).
  - **Dynamic Lighting Simulation:** Apply realistic overlay masks simulating daylight, warm light, moon light, and sunset.
  - **XAI Modals:** Transparent, user-friendly modals rendering real-time progress bars and score breakdowns explaining exactly *why* an item was recommended.
  - **Exporting:** Integrated `html2canvas` for downloading final designs.

### 2. Backend (Flask Server)
- **Stack:** Python, Flask, Flask-CORS.
- **Image Processing Engine:** OpenCV (`cv2`), `scikit-image`.
- **Deep Learning/ML:** TensorFlow/Keras (`ResNet50`), `scikit-learn` (K-Means).
- **Background Removal pipeline:** `rembg` (U²-Net architecture).
- **Core Endpoints:**
  - `/ai-recommend`: Accepts a room image upload, extracts 8 sets of features, and calculates weighted recommendation scores against the entire dataset.
  - `/recommend`: Text-based keyword filtering followed by ML parameter ranking.
  - `/prepare`: Real-time AI background removal pipeline generating transparent `.png` buffers for seamless canvas placement.
  - `/image/<filename>`: Static asset serving.

---

## 🧠 Machine Learning & AI Core

VisionHome moves beyond simple keyword matching by implementing a robust, multi-parameter scoring algorithm. Every furniture item is ranked out of 100% based on the following weighted criteria:

1. **Visual Feature Similarity (40% - Deep Learning)**
   - Utilizes a pre-trained **ResNet50** Convolutional Neural Network (CNN) as a feature extractor.
   - The top classification layer is removed, generating a **2048-dimensional embedding vector**.
   - The system calculates the cosine similarity between the uploaded room image vector and the dataset furniture vectors.
   
2. **Color Compatibility (15% - Unsupervised ML)**
   - Implements **K-Means Clustering** (where k=3) on the image's RGB pixel space to mathematically extract the dominant color palette centroids.
   - Calculates the minimum Euclidean distance between the room's dominant colors and the furniture's dominant colors.

3. **Color Distribution Similarity (10% - Computer Vision)**
   - Extracts a normalized 2D **HSV (Hue-Saturation-Value) Histogram** (8x8 bins).
   - Computes the intersection of histograms between the room and the item.

4. **Brightness Compatibility (10%)**
   - Extracts grayscale mean intensity (0.0 to 1.0) and penalizes items that drastically differ from the ambient lighting of the uploaded room.

5. **Texture Similarity (10% - Computer Vision)**
   - Extracts a micro-texture descriptor using **LBP (Local Binary Pattern)**.
   - Generates a 10-bin uniform histogram of textures and calculates the histogram intersection to ensure, for example, heavily textured rooms are matched with appropriately textured furniture.

6. **Shape/Edge Similarity (5%)**
   - Applies a **Canny Edge Detector** to calculate edge density and structural complexity.
   - Generates an 8-bin orientation histogram using Sobel derivatives to match structural flow (e.g., matching sharp geometric rooms with angular furniture).

7. **Visual Style Feature (5%)**
   - An encoded heuristic vector combining normalized brightness, contrast, saturation, and texture variance.

8. **Category Relevance (5%)**
   - A metadata-driven check to maintain logical parsing.

*(Note: In order to preserve rapid frontend performance, dataset metadata is pre-computed and stored in `dataset_metadata.pkl` using the `precompute_ml_features.py` pipeline.)*

---

## 📊 Explainable AI (XAI)

A critical component of VisionHome is its transparency. Often, recommendation systems act as "black boxes." VisionHome breaks this paradigm by offering an **XAI — Explainable Recommendations** flow. 

When a user clicks "Why Recommended?" on any furniture card, the frontend queries the exact mathematical breakdown of the 8 parameters described above. It exposes the ResNet50 similarity scores, the K-Means color distance, and all localized CV metrics in easy-to-read progress bars, ensuring users understand that recommendations are grounded in complex mathematical relationships, not random assignments.

---

## 🚀 How to Run Locally

Start the Backend and Frontend servers simultaneously in two separate terminals.

**Terminal 1 (Backend):**
```bash
cd flask_backend
source venv/bin/activate  # (If using a virtual environment)
python app.py
```
*(Server will start on http://127.0.0.1:5050)*

**Terminal 2 (Frontend):**
```bash
cd frontend
python -m http.server 8000
```
*(Application will be available at http://127.0.0.1:8000)*