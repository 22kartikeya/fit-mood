import os
import pandas as pd
from PIL import Image

# Load metadata
df = pd.read_csv("fitmood_metadata.csv")
print(df.head())

original_img_dir = "./downloaded_images"
processed_img_dir = "processed_images/"

# Make sure processed folder exists
os.makedirs(processed_img_dir, exist_ok=True)

# Get list of actual image files in lowercase
available_files = set(f.lower() for f in os.listdir(original_img_dir))

# Image size
img_size = (224, 224)

# Process images
missing = []
for i, row in df.iterrows():
    image_file = row["image"]
    if image_file.lower() in available_files:
        try:
            img_path = os.path.join(original_img_dir, image_file)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)
            img.save(os.path.join(processed_img_dir, image_file))
        except:
            missing.append(image_file)
    else:
        missing.append(image_file)

print(f"✅ Done! Processed images saved to: {processed_img_dir}")
if missing:
    print(f"⚠️ Missing or corrupted images: {len(missing)}")