import os
import urllib.request
from duckduckgo_search import DDGS

def download_semantic_samples():
    os.makedirs("dataset/semantic_images", exist_ok=True)
    queries = {
        "person": "person walking full body clear photography",
        "mobile": "person holding cell phone close up",
        "clock": "wall clock white background photography",
        "laptop": "laptop on a desk photography"
    }

    print("Downloading semantic object images using DuckDuckGo...")
    
    count = 1
    with DDGS() as ddgs:
        for category, query in queries.items():
            print(f"Searching for {category}...")
            results = ddgs.images(query, max_results=15)
            downloaded_for_cat = 0
            
            for r in results:
                if downloaded_for_cat >= 12:
                    break
                url = r.get("image")
                if not url: continue
                
                filepath = f"dataset/semantic_images/{category}_{count:03d}.jpg"
                try:
                    # add user agent to avoid 403
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=5) as response, open(filepath, 'wb') as out_file:
                        out_file.write(response.read())
                    print(f"Downloaded {filepath}")
                    count += 1
                    downloaded_for_cat += 1
                except Exception as e:
                    pass
            
    print(f"Successfully downloaded {count-1} semantic sample images to dataset/semantic_images/")

if __name__ == "__main__":
    download_semantic_samples()
