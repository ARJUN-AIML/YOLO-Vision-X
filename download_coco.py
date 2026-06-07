import os
import urllib.request
import time

def download_coco_images():
    os.makedirs("dataset/semantic_images", exist_ok=True)
    print("Downloading images from COCO val2017...")
    
    downloaded = 0
    # Try IDs sequentially; COCO val2017 has IDs spread out, so we might have some 404s, but we'll find 50.
    # Actually, we can use a known list of COCO val2017 IDs or just search
    # Let's try some known IDs or just brute force a range where we know there are images.
    # A safer bet is to use random IDs up to 600,000.
    
    import random
    random.seed(42)
    attempts = 0
    
    while downloaded < 50 and attempts < 1000:
        # COCO IDs are up to 12 digits, but typically 6 digits in val2017.
        # Let's try 397133, 000139, 000285, 000632, etc.
        # Actually, let's just grab COCO train2014 or val2017 IDs from a small public gist!
        break
        
def download_from_gist():
    # Let's just download 50 images from picsum again, wait, the user said picsum didn't have person/mobile/clock!
    pass

# Instead of random guessing, I'll download a few specific images that I know the URL of from COCO.
urls = [
    "http://images.cocodataset.org/val2017/000000397133.jpg", # Person, kitchen
    "http://images.cocodataset.org/val2017/000000037777.jpg", # Person
    "http://images.cocodataset.org/val2017/000000252219.jpg", # Person
    "http://images.cocodataset.org/val2017/000000087038.jpg", # Person
    "http://images.cocodataset.org/val2017/000000174482.jpg", # Person
    "http://images.cocodataset.org/val2017/000000400872.jpg", # Person
    "http://images.cocodataset.org/val2017/000000000139.jpg", # Person
    "http://images.cocodataset.org/val2017/000000000285.jpg", # Bear
    "http://images.cocodataset.org/val2017/000000000632.jpg", # Bedroom
    "http://images.cocodataset.org/val2017/000000000776.jpg", # Teddy bears
    "http://images.cocodataset.org/val2017/000000000802.jpg", # Oven
    "http://images.cocodataset.org/val2017/000000000872.jpg", # Person playing baseball
]

def download_specific_coco():
    os.makedirs("dataset/semantic_images", exist_ok=True)
    count = 1
    for url in urls:
        filepath = f"dataset/semantic_images/coco_{count:03d}.jpg"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Downloaded {url}")
            count += 1
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            
if __name__ == "__main__":
    download_specific_coco()
