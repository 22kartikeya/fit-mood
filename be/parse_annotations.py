import json
import os
import pandas as pd

annotation_dir = "./annotations/train/"
metadata = []

for filename in os.listdir(annotation_dir):
    if filename.endswith(".json"):
        with open(os.path.join(annotation_dir, filename), "r") as f:
            data = json.load(f)
            for item in data["item1"]:
                metadata.append({
                    "image": data["image_name"],
                    "category": item.get("category_name", "Unknown"),
                    "style": "Casual",  # You can modify this manually or label later
                    "color": "Unknown"  # Optional: Can extract from file name or manually
                })

# Save to CSV
df = pd.DataFrame(metadata)
df.to_csv("fitmood_metadata.csv", index=False)

print("✅ Metadata parsed and saved to fitmood_metadata.csv")