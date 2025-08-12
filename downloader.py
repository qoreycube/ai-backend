from bing_image_downloader import downloader
from fastai.vision.all import *
from ddgs import DDGS
import requests
import os
from pathlib import Path

# Read bird species from file
def load_bird_species(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Get the directory where this script is located
script_dir = Path(__file__).parent
bird_species = load_bird_species(script_dir / 'bird_species.txt')

# Root folder for all assets
base_dir = script_dir / 'assets' / 'ddgbirds'

def sanitize_folder_name(name):
    # Replace spaces with underscores and remove special characters
    return name.replace(" ", "_").replace("-", "_").lower()

def download_images_bing(query, folder, limit=100):  # Increased from 50 to 100
    dest = base_dir/folder
    if dest.exists() and any(dest.glob('*.jpg')):
        print(f"✅ Skipping {folder}, already has images.")
        return

    print(f"⬇️  Downloading: {query}")
    try:
        downloader.download(
            query,
            limit=limit,
            output_dir=str(base_dir),
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )

        # Rename auto-generated folder to match our folder name
        downloaded = base_dir / query
        if downloaded.exists():
            downloaded.rename(dest)
        else:
            print(f"⚠️  Warning: Downloaded folder not found for {query}")
    except Exception as e:
        print(f"❌ Error downloading {query}: {str(e)}")

def download_images_duckduckgo(query, folder, limit=50):
    dest = base_dir/folder
    if dest.exists() and any(dest.glob('*.jpg')):
        print(f"✅ Skipping {folder}, already has images.")
        return

    os.makedirs(dest, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=limit)
        for i, result in enumerate(results):
            url = result['image']
            ext = url.split('.')[-1].split('?')[0]
            filename = os.path.join(dest, f"{query.replace(' ', '_')}_{i}.{ext}")
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    print(f"Downloaded: {filename}")
                    with open(filename, 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

# Step 1: Download images (only if 'download' argument is passed)
print(f"📥 Starting download for {len(bird_species)} bird species...")
for i, bird in enumerate(bird_species, 1):
    folder_name = sanitize_folder_name(bird)
    print(f"[{i}/{len(bird_species)}] Processing: {bird}")
    download_images_duckduckgo(f"{bird} bird", folder_name)
print("✅ Download phase completed!")

# Step 1.5: Clean up downloaded images (remove corrupted/invalid files)
print("🧹 Cleaning up downloaded images...")
def clean_images(path):
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f"Removed {len(failed)} corrupted images")

clean_images(base_dir)

