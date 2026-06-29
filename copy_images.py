import os
import shutil

# Source: dataset images
source_folder = "furniture-detector/img"  # change if your dataset is elsewhere

# Destination: frontend assets
dest_folder = "frontend/assets/furniture"
os.makedirs(dest_folder, exist_ok=True)

# Copy all images
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dest_folder, file)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file}")

print("All images copied successfully!")

