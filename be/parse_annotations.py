import json
import os
import pandas as pd

# Load your existing styles.csv
try:
    df = pd.read_csv("styles.csv")  # Replace with your CSV filename if different
except FileNotFoundError:
    print("❌ styles.csv not found. Make sure it's in the same directory as this script.")
    exit()

# Clean and format metadata
metadata = pd.DataFrame({
    "image": df["id"].astype(str) + ".jpg",                     # Construct image filename
    "category": df["articleType"],                              # Clothing category
    "style": df["usage"].fillna("Casual"),                      # Usage (fallback = Casual)
    "color": df["baseColour"].fillna("Unknown")                # Color (fallback = Unknown)
})

# Drop entries with missing image or category
metadata = metadata.dropna(subset=["image", "category"])

# Save to a new CSV file
metadata.to_csv("fitmood_metadata.csv", index=False)

print("✅ Metadata parsed and saved to fitmood_metadata.csv")