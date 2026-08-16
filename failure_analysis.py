"""
failure_analysis.py
---------------------
DAY 8 (continued): Finds every development query photo where the
BASELINE model's top match was WRONG, and organizes them into a
folder so you can visually inspect what went wrong.

WHAT THIS DOES:
- Goes through every catalogue development query photo.
- If the #1 (top) match was NOT the correct category, it's a "failure".
- For each failure, copies TWO images into failure_cases/:
    1. The query photo itself (what was uploaded)
    2. The gallery image it was WRONGLY matched to
  ...named so they're easy to compare side by side.
- Also writes failure_log.csv with the details (true category, wrong
  category it matched, similarity score) and an empty "reason" column
  for you to fill in after looking at each pair.

HOW TO RUN (from inside catalogue_project folder):
    python failure_analysis.py

REQUIRES:
    gallery_embeddings.npz
    query_split/development/
    gallery/  (original folder, to copy example images from)

AFTER RUNNING:
    Open the failure_cases/ folder and look at each pair of images.
    Common reasons for failure (write these in the "reason" column):
        - blurry / motion blur
        - unusual angle
        - poor lighting / glare
        - item partially hidden or too small in frame
        - genuinely similar-looking item (e.g. two different pens)
        - background clutter confusing the model
"""

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import csv

GALLERY_EMB_FILE = "gallery_embeddings.npz"
GALLERY_FOLDER = "gallery"
DEV_QUERY_FOLDER = os.path.join("query_split", "development")
NOT_IN_CATALOGUE_FOLDER_NAME = "not_in_catalogue"
OUTPUT_FOLDER = "failure_cases"
LOG_FILE = "failure_log.csv"

VALID_EXTENSIONS = (
    ".jpg", ".jpeg", ".jfif", ".jpe", ".jif",
    ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif", ".avif",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading pretrained ResNet50 model (baseline)...")
weights = models.ResNet50_Weights.IMAGENET1K_V2
base_model = models.resnet50(weights=weights)
base_model.fc = nn.Identity()
base_model.eval()
base_model.to(device)
preprocess = weights.transforms()


def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = base_model(tensor)
    return emb.squeeze().cpu().numpy()


def cosine_sim(query_vec, all_vecs):
    q = query_vec / np.linalg.norm(query_vec)
    a = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return a @ q


def main():
    data = np.load(GALLERY_EMB_FILE, allow_pickle=True)
    gallery_embeddings = data["embeddings"]
    gallery_labels = data["labels"]
    gallery_filenames = data["filenames"]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    categories = sorted(
        f for f in os.listdir(DEV_QUERY_FOLDER)
        if os.path.isdir(os.path.join(DEV_QUERY_FOLDER, f))
    )

    failures = []
    case_number = 0

    print("Scanning development query photos for wrong matches...\n")

    for category in categories:
        if category == NOT_IN_CATALOGUE_FOLDER_NAME:
            continue

        folder_path = os.path.join(DEV_QUERY_FOLDER, category)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)]

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                query_emb = get_embedding(fpath)
            except Exception:
                continue

            sims = cosine_sim(query_emb, gallery_embeddings)
            best_idx = np.argmax(sims)
            predicted_label = gallery_labels[best_idx]
            score = sims[best_idx]

            if predicted_label != category:
                case_number += 1

                query_ext = os.path.splitext(fname)[1]
                query_dest = os.path.join(
                    OUTPUT_FOLDER,
                    f"case{case_number:02d}_QUERY_actual-{category}{query_ext}"
                )
                shutil.copy2(fpath, query_dest)

                gallery_src = os.path.join(GALLERY_FOLDER, predicted_label, gallery_filenames[best_idx])
                if os.path.exists(gallery_src):
                    gallery_ext = os.path.splitext(gallery_filenames[best_idx])[1]
                    gallery_dest = os.path.join(
                        OUTPUT_FOLDER,
                        f"case{case_number:02d}_MATCHED-{predicted_label}_score{score:.2f}{gallery_ext}"
                    )
                    shutil.copy2(gallery_src, gallery_dest)

                failures.append({
                    "case": case_number,
                    "query_file": fname,
                    "true_category": category,
                    "wrongly_matched_to": predicted_label,
                    "similarity_score": round(float(score), 3),
                    "reason": "",
                })

        print(f"  Scanned: {category}")

    print(f"\nTotal failures found: {len(failures)}")

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case", "query_file", "true_category", "wrongly_matched_to",
            "similarity_score", "reason"
        ])
        writer.writeheader()
        writer.writerows(failures)

    print(f"Saved failure details to '{LOG_FILE}'.")
    print(f"Saved image pairs to '{OUTPUT_FOLDER}/' for visual inspection.")
    print("\nNext step: open the failure_cases folder, look at each")
    print("case# pair of images, and fill in the 'reason' column in")
    print(f"{LOG_FILE} (e.g. blurry, wrong angle, similar-looking item).")


if __name__ == "__main__":
    main()