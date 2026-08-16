"""
score_baseline.py
-------------------
DAY 4 (real evaluation): Tests how well the BASELINE (plain pretrained
ResNet50, no extra training) performs on your actual development query
photos - the messy, real-world phone photos.

WHAT IT MEASURES (matches the project's required measurements):
1. Recall@1  - how often the TRUE category is the #1 (top) match
2. Recall@5  - how often the TRUE category appears anywhere in the top 5
3. False matches on "not_in_catalogue" photos - how often an unlisted
   object still gets matched to something (instead of being refused)
   -- NOTE: at this baseline stage we don't have a cut-off/threshold
   yet (that comes on Day 7), so this just reports the average
   similarity score for not_in_catalogue photos vs catalogue photos,
   which will help us CHOOSE a good cut-off later.

HOW TO RUN (from inside catalogue_project folder):
    python score_baseline.py

REQUIRES:
    gallery_embeddings.npz (already created)
    query_split/development/ (already created)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

GALLERY_EMB_FILE = "gallery_embeddings.npz"
DEV_QUERY_FOLDER = os.path.join("query_split", "development")
NOT_IN_CATALOGUE_FOLDER_NAME = "not_in_catalogue"

VALID_EXTENSIONS = (
    ".jpg", ".jpeg", ".jfif", ".jpe", ".jif",
    ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif", ".avif",
)

print("Loading pretrained ResNet50 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        embedding = base_model(tensor)
    return embedding.squeeze().cpu().numpy()


def cosine_sim(query_vec, all_vecs):
    q = query_vec / np.linalg.norm(query_vec)
    a = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return a @ q


def main():
    data = np.load(GALLERY_EMB_FILE, allow_pickle=True)
    gallery_embeddings = data["embeddings"]
    gallery_labels = data["labels"]
    print(f"Loaded {gallery_embeddings.shape[0]} gallery embeddings.\n")

    if not os.path.isdir(DEV_QUERY_FOLDER):
        print(f"Could not find '{DEV_QUERY_FOLDER}'.")
        return

    categories = sorted(
        f for f in os.listdir(DEV_QUERY_FOLDER)
        if os.path.isdir(os.path.join(DEV_QUERY_FOLDER, f))
    )

    top1_correct = 0
    top5_correct = 0
    total_catalogue_queries = 0

    not_in_catalogue_scores = []
    catalogue_best_scores = []

    print("Scoring development query photos...\n")

    for category in categories:
        folder_path = os.path.join(DEV_QUERY_FOLDER, category)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)]

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                query_emb = get_embedding(fpath)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                continue

            sims = cosine_sim(query_emb, gallery_embeddings)
            top5_idx = np.argsort(sims)[::-1][:5]
            top5_labels = gallery_labels[top5_idx]
            best_score = sims[top5_idx[0]]

            if category == NOT_IN_CATALOGUE_FOLDER_NAME:
                not_in_catalogue_scores.append(best_score)
            else:
                total_catalogue_queries += 1
                catalogue_best_scores.append(best_score)
                if top5_labels[0] == category:
                    top1_correct += 1
                if category in top5_labels:
                    top5_correct += 1

        print(f"  Done: {category} ({len(files)} photos)")

    print("\n" + "=" * 50)
    print("BASELINE RESULTS (development set)")
    print("=" * 50)

    if total_catalogue_queries > 0:
        recall1 = top1_correct / total_catalogue_queries
        recall5 = top5_correct / total_catalogue_queries
        print(f"\nCatalogue queries tested: {total_catalogue_queries}")
        print(f"Recall@1: {recall1:.1%}  ({top1_correct}/{total_catalogue_queries})")
        print(f"Recall@5: {recall5:.1%}  ({top5_correct}/{total_catalogue_queries})")

    if catalogue_best_scores:
        print(f"\nCatalogue photos - similarity score to their best match:")
        print(f"  average: {np.mean(catalogue_best_scores):.3f}")
        print(f"  minimum: {np.min(catalogue_best_scores):.3f}")

    if not_in_catalogue_scores:
        print(f"\nNot-in-catalogue photos - similarity score to their best (wrong) match:")
        print(f"  average: {np.mean(not_in_catalogue_scores):.3f}")
        print(f"  maximum: {np.max(not_in_catalogue_scores):.3f}")
        print("\n(We'll use these two score distributions on Day 7 to pick a")
        print(" cut-off threshold that separates real matches from false ones.)")


if __name__ == "__main__":
    main()