import os
import pandas as pd
import requests
from tqdm import tqdm  # For progress bar

# Load the CSV with custom filenames and URLs
df = pd.read_csv("images.csv", header=None, names=["filename", "url"])

# Create folder to save images
download_dir = "downloaded_images/"
os.makedirs(download_dir, exist_ok=True)

# Download images
for _, row in tqdm(df.iterrows(), total=len(df), desc="📥 Downloading images"):
    try:
        filename = row["filename"]
        url = row["url"]
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(os.path.join(download_dir, filename), "wb") as f:
                f.write(response.content)
        else:
            print(f"❌ Failed to download ({response.status_code}): {url}")
    except Exception as e:
        print(f"⚠ Error downloading {url}: {e}")

print("✅ All downloads attempted.")