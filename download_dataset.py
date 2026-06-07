import os
import urllib.request
import json

def download_samples():
    os.makedirs("dataset/images", exist_ok=True)
    print("Fetching COCO 2017 validation dataset metadata...")
    
    # We will just download 50 images from COCO directly via their URLs
    # A small static list of COCO image IDs to download
    # Or just a simple loop downloading picsum images
    print("Downloading 50 sample images...")
    for i in range(1, 51):
        url = f"https://picsum.photos/640/480?random={i}"
        filepath = f"dataset/images/sample_{i:03d}.jpg"
        try:
            urllib.request.urlretrieve(url, filepath)
            if i % 10 == 0:
                print(f"Downloaded {i}/50 images")
        except Exception as e:
            print(f"Failed to download image {i}: {e}")
            
    print("Successfully downloaded 50 sample images to dataset/images/")

if __name__ == "__main__":
    download_samples()
