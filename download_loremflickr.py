import os
import urllib.request
import time

def download_loremflickr():
    os.makedirs("dataset/semantic_images", exist_ok=True)
    categories = ["person", "mobile", "clock", "laptop", "car"]
    print("Downloading semantic object images from LoremFlickr...")
    
    count = 1
    for cat in categories:
        for i in range(10): # 10 per category = 50 total
            url = f"https://loremflickr.com/640/480/{cat}?lock={i}"
            filepath = f"dataset/semantic_images/{cat}_{i:03d}.jpg"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"Downloaded {cat} image {i+1}/10")
                count += 1
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed to download {cat} image {i+1}: {e}")
                
    print(f"Successfully downloaded {count-1} semantic sample images to dataset/semantic_images/")

if __name__ == "__main__":
    download_loremflickr()
