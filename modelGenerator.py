from bing_image_downloader import downloader
from fastai.vision.all import *
import os
import sys
from pathlib import Path

# Read bird species from file
def load_bird_species(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Get the directory where this script is located
script_dir = Path(__file__).parent
bird_species = load_bird_species(script_dir / 'bird_species.txt')

# Root folder for all assets
base_dir = script_dir / 'assets' / 'birds'

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


# Step 1: Download images (only if 'download' argument is passed)
if 'download' in sys.argv:
    print(f"📥 Starting download for {len(bird_species)} bird species...")
    for i, bird in enumerate(bird_species, 1):
        folder_name = sanitize_folder_name(bird)
        print(f"[{i}/{len(bird_species)}] Processing: {bird}")
        download_images_bing(f"{bird} bird", folder_name)
    print("✅ Download phase completed!")

# Step 1.5: Clean up downloaded images (remove corrupted/invalid files)
print("🧹 Cleaning up downloaded images...")
def clean_images(path):
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    print(f"Removed {len(failed)} corrupted images")

clean_images(base_dir)

# Step 2: Create DataLoaders with enhanced augmentation
dls = ImageDataLoaders.from_folder(
    base_dir,
    valid_pct=0.2,
    seed=42,
    item_tfms=[Resize(460)],  # Larger initial size for better crops
    batch_tfms=[
        *aug_transforms(
            size=224,
            min_scale=0.75,  # More aggressive cropping
            do_flip=True,
            flip_vert=False,  # Birds shouldn't be upside down
            max_rotate=15.0,  # Slight rotation
            max_zoom=1.2,
            max_lighting=0.3,
            max_warp=0.2,
            p_affine=0.8,
            p_lighting=0.8
        ),
        Normalize.from_stats(*imagenet_stats)
    ]#,
#    bs=2
)

# Optional: Show sample images
#dls.show_batch(max_n=9)

# Step 3: Train the model with improved techniques
# Use a more powerful architecture
print("🏋️  Creating vision_learner")

learn = vision_learner(
    dls, 
    resnet50,  # Reduced from resnet34 or resnet50
    metrics=[accuracy, partial(top_k_accuracy, k=min(5, dls.c))],  # Use partial to set k
    pretrained=True
)

learn.to_fp16()

# Find optimal learning rate
print("🏋️  Running lr_find()")
learn.lr_find()

# Progressive training approach
print("🏋️  Starting progressive training...")

# Phase 1: Train head only with higher learning rate
print("Phase 1: Training classifier head...")
learn.freeze()
learn.fit_one_cycle(3, lr_max=slice(1e-3))

# Phase 2: Unfreeze and train with discriminative learning rates
print("Phase 2: Fine-tuning all layers...")
learn.unfreeze()
learn.fit_one_cycle(
    8,  # More epochs
    lr_max=slice(1e-6, 1e-4),  # Discriminative learning rates
    div=25,
    pct_start=0.8  # Longer warmup
)

# Phase 3: Additional fine-tuning with lower learning rate
print("Phase 3: Final polish training...")
learn.fit_one_cycle(
    4, 
    lr_max=slice(1e-7, 5e-5),
    div=25
)

# Step 3.5: Evaluate model performance
print("📊 Evaluating model performance...")
learn.show_results(max_n=16)

# Show confusion matrix for detailed analysis
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(20, 20), dpi=60)

# Show top losses (hardest to classify images)
interp.plot_top_losses(9, nrows=3)

# Print final accuracy
valid_accuracy = learn.validate()[1]
print(f"🎯 Final Validation Accuracy: {valid_accuracy:.4f} ({valid_accuracy*100:.2f}%)")

# Test Time Augmentation for improved inference accuracy
print("🔮 Testing with Test Time Augmentation (TTA)...")
#tta_accuracy = learn.tta()[1]
#tta_acc_value = tta_accuracy.item() if hasattr(tta_accuracy, 'item') else float(tta_accuracy)
#print(f"🚀 TTA Validation Accuracy: {tta_acc_value:.4f} ({tta_acc_value*100:.2f}%)")

# Step 4: Export trained model
model_filename = f'bird_classifier_{len(bird_species)}_species.pkl'
learn.export(model_filename)
print(f"🎯 Model exported as: {model_filename}")