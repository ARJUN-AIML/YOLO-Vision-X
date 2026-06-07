import os
import urllib.request

def download_static_images():
    os.makedirs("dataset/semantic_images", exist_ok=True)
    urls = {
        "person_001.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/38/Two_people_walking.jpg",
        "person_002.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/e1/A_person_walking_in_the_snow.jpg",
        "person_003.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b5/Man_walking_in_the_street.jpg",
        "mobile_001.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b4/Smartphone_in_hand.jpg",
        "mobile_002.jpg": "https://upload.wikimedia.org/wikipedia/commons/f/ff/Smartphone.jpg",
        "clock_001.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/75/Clock-2.jpg",
        "clock_002.jpg": "https://upload.wikimedia.org/wikipedia/commons/8/87/Wall_clock.jpg",
        "laptop_001.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/43/Laptop_on_a_desk.jpg"
    }

    print("Downloading semantic object images from Wikimedia Commons...")
    
    count = 0
    for filename, url in urls.items():
        filepath = f"dataset/semantic_images/{filename}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Downloaded {filepath}")
            count += 1
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            
    print(f"Successfully downloaded {count} semantic sample images to dataset/semantic_images/")

if __name__ == "__main__":
    download_static_images()
