"""
choose_cutoff.py
------------------
DAY 7: Finds a good similarity cut-off (rejection threshold) using the
BASELINE model (since baseline outperformed the trained projection on
our development set) and the development query photos.

WHAT THIS DOES, IN SIMPLE WORDS:
- For every development query photo, we already know its best similarity
  score to the gallery (computed the same way as score_baseline.py).
- We try many possible threshold values (e.g. 0.30, 0.31, 0.32 ... 0.60).
- For EACH threshold, we measure two kinds of mistakes:
    1. False refusals: real catalogue items wrongly rejected
       (their score fell below the threshold)
    2. False matches: not-in-catalogue items wrongly accepted
       (their score was at or above the threshold)
- We print a small table so you can see the trade-off, and recommend
  the threshold where these two error rates are closest to equal
  (a common, defensible choice called the "equal error rate" point).
- We also save a plot (cutoff_curve.png) showing this trade-off - this
  plot is exactly what your report needs for the "curve" requirement.

HOW TO RUN (from inside catalogue_project folder):
    python choose_cutoff.py

REQUIRES:
    gallery_embeddings.npz
    query_split/development/
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

GALLERY_EMB_FILE = "gallery_embeddings.npz"
DEV_QUERY_FOLDER = os.path.join("query_split", "development")
NOT_IN_CATALOGUE_FOLDER_NAME = "not_in_catalogue"
PLOT_OUTPUT_FILE = "cutoff_curve.png"

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

    categories = sorted(
        f for f in os.listdir(DEV_QUERY_FOLDER)
        if os.path.isdir(os.path.join(DEV_QUERY_FOLDER, f))
    )

    catalogue_scores = []
    not_in_catalogue_scores = []

    print("Collecting similarity scores for every development query photo...\n")
    for category in categories:
        folder_path = os.path.join(DEV_QUERY_FOLDER, category)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)]

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                query_emb = get_embedding(fpath)
            except Exception:
                continue
            sims = cosine_sim(query_emb, gallery_embeddings)
            best_score = sims.max()

            if category == NOT_IN_CATALOGUE_FOLDER_NAME:
                not_in_catalogue_scores.append(best_score)
            else:
                catalogue_scores.append(best_score)

        print(f"  Done: {category}")

    catalogue_scores = np.array(catalogue_scores)
    not_in_catalogue_scores = np.array(not_in_catalogue_scores)

    print(f"\nCatalogue queries: {len(catalogue_scores)}")
    print(f"Not-in-catalogue queries: {len(not_in_catalogue_scores)}\n")

    thresholds = np.arange(0.20, 0.70, 0.01)
    false_refusal_rates = []
    false_match_rates = []

    print(f"{'Threshold':>10} | {'False Refusals':>15} | {'False Matches':>14}")
    print("-" * 46)

    best_threshold = None
    best_gap = float("inf")

    for t in thresholds:
        false_refusals = (catalogue_scores < t).mean()
        false_matches = (not_in_catalogue_scores >= t).mean()
        false_refusal_rates.append(false_refusals)
        false_match_rates.append(false_matches)

        gap = abs(false_refusals - false_matches)
        if gap < best_gap:
            best_gap = gap
            best_threshold = t

        if round(t * 100) % 5 == 0:
            print(f"{t:>10.2f} | {false_refusals:>14.1%} | {false_matches:>13.1%}")

    print("\n" + "=" * 50)
    print(f"RECOMMENDED CUT-OFF (equal error rate point): {best_threshold:.2f}")
    print("=" * 50)
    print("At this threshold:")
    fr = (catalogue_scores < best_threshold).mean()
    fm = (not_in_catalogue_scores >= best_threshold).mean()
    print(f"  False refusals on catalogue items: {fr:.1%}")
    print(f"  False matches on not-in-catalogue items: {fm:.1%}")

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, false_refusal_rates, label="False refusals (valid items rejected)")
    plt.plot(thresholds, false_match_rates, label="False matches (unlisted objects accepted)")
    plt.axvline(best_threshold, color="gray", linestyle="--", label=f"Chosen cut-off = {best_threshold:.2f}")
    plt.xlabel("Similarity threshold")
    plt.ylabel("Error rate")
    plt.title("Cut-off trade-off curve (development set, baseline model)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_FILE, dpi=150)
    print(f"\nSaved trade-off curve plot to '{PLOT_OUTPUT_FILE}'.")
    print("Include this plot in your report - it satisfies the 'plot the two")
    print("kinds of refusal against each other' requirement from Day 7.")


if __name__ == "__main__":
    main()