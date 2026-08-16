"""
extract_embeddings.py
-----------------------
DAY 4 SCRIPT: Turns every gallery image into a numeric "fingerprint"
(an embedding vector) using a pretrained ResNet50 model, and saves
all the fingerprints into one file for later use.

WHAT THIS DOES, IN SIMPLE WORDS:
- Loads a pretrained image model (ResNet50) that was already trained
  on millions of images. We remove its final classification layer,
  so instead of giving a "class label" it gives us a vector of
  numbers describing what's in the image.
- Runs every image in catalogue_project/gallery/ through this model.
- Saves all the resulting vectors + which item/category each one
  belongs to, into a single file: gallery_embeddings.npz

This is the BASELINE step (Day 4 in the work plan). Later (Day 6)
we will train a small extra layer on top of this to make it more
accurate specifically for your 26 items.

HOW TO RUN:
    python extract_embeddings.py

REQUIREMENTS (already installed on your machine):
    torch, torchvision, numpy, pillow
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

GALLERY_FOLDER = "gallery"
OUTPUT_FILE = "gallery_embeddings.npz"
VALID_EXTENSIONS = (
    ".jpg", ".jpeg", ".jfif", ".jpe", ".jif",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif", ".tiff",
    ".heic", ".heif",
    ".avif",
)

# ---------------------------------------------------------
# STEP 1: Load the pretrained model, remove its classification head
# ---------------------------------------------------------
print("Loading pretrained ResNet50 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

weights = models.ResNet50_Weights.IMAGENET1K_V2
base_model = models.resnet50(weights=weights)

# Remove the final classification layer (fc).
# What remains outputs a 2048-number vector per image - our "fingerprint".
base_model.fc = nn.Identity()
base_model.eval()
base_model.to(device)

# The exact resize/crop/normalize steps the pretrained model expects
preprocess = weights.transforms()


def get_embedding(image_path):
    """Loads one image and returns its embedding vector as a numpy array."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)  # add batch dimension
    with torch.no_grad():
        embedding = base_model(tensor)
    return embedding.squeeze().cpu().numpy()


# ---------------------------------------------------------
# STEP 2: Go through every image in every gallery folder
# ---------------------------------------------------------
def main():
    if not os.path.isdir(GALLERY_FOLDER):
        print(f"Could not find {GALLERY_FOLDER}.")
        print("Make sure you run this script from the same folder as catalogue_project.")
        return

    categories = sorted(
        f for f in os.listdir(GALLERY_FOLDER)
        if os.path.isdir(os.path.join(GALLERY_FOLDER, f))
    )
    print(f"Found {len(categories)} categories.\n")

    all_embeddings = []
    all_labels = []       # which category each embedding belongs to
    all_filenames = []    # original filename, useful for debugging later

    for category in categories:
        folder_path = os.path.join(GALLERY_FOLDER, category)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)]
        print(f"Processing {category}: {len(files)} images")

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                emb = get_embedding(fpath)
            except Exception as e:
                print(f"  Skipping {fname} (could not process): {e}")
                continue
            all_embeddings.append(emb)
            all_labels.append(category)
            all_filenames.append(fname)

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)
    all_filenames = np.array(all_filenames)

    print(f"\nTotal embeddings created: {all_embeddings.shape[0]}")
    print(f"Each embedding has {all_embeddings.shape[1]} numbers.")

    np.savez(
        OUTPUT_FILE,
        embeddings=all_embeddings,
        labels=all_labels,
        filenames=all_filenames,
    )
    print(f"\nSaved everything to '{OUTPUT_FILE}'.")
    print("This file will be reused in the next steps - no need to redo this again.")


if __name__ == "__main__":
    main()