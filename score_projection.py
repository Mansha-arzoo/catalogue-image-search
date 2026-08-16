"""
score_projection.py
---------------------
Tests the TRAINED PROJECTION (from train_projection.py) on the same
development query photos used for the baseline, so we can directly
compare: did training the projection actually help?

HOW TO RUN (from inside catalogue_project folder):
    python score_projection.py

REQUIRES:
    projection_weights.pth
    gallery_embeddings_projected.npz
    query_split/development/
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

PROJECTED_GALLERY_FILE = "gallery_embeddings_projected.npz"
PROJECTION_WEIGHTS_FILE = "projection_weights.pth"
DEV_QUERY_FOLDER = os.path.join("query_split", "development")
NOT_IN_CATALOGUE_FOLDER_NAME = "not_in_catalogue"

INPUT_DIM = 2048
HIDDEN_DIM = 512
PROJECTED_DIM = 256

VALID_EXTENSIONS = (
    ".jpg", ".jpeg", ".jfif", ".jpe", ".jif",
    ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif", ".avif",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


print("Loading pretrained ResNet50 (feature extractor)...")
weights = models.ResNet50_Weights.IMAGENET1K_V2
base_model = models.resnet50(weights=weights)
base_model.fc = nn.Identity()
base_model.eval()
base_model.to(device)
preprocess = weights.transforms()

print("Loading trained projection layer...")
projection = ProjectionHead(INPUT_DIM, HIDDEN_DIM, PROJECTED_DIM).to(device)
projection.load_state_dict(torch.load(PROJECTION_WEIGHTS_FILE, map_location=device))
projection.eval()


def get_projected_embedding(image_path):
    """Runs an image through ResNet50, then through the trained projection."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        raw_emb = base_model(tensor)
        projected_emb = projection(raw_emb)
    return projected_emb.squeeze().cpu().numpy()


def cosine_sim(query_vec, all_vecs):
    q = query_vec / np.linalg.norm(query_vec)
    a = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return a @ q


def main():
    data = np.load(PROJECTED_GALLERY_FILE, allow_pickle=True)
    gallery_embeddings = data["embeddings"]
    gallery_labels = data["labels"]
    print(f"Loaded {gallery_embeddings.shape[0]} projected gallery embeddings.\n")

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

    print("Scoring development query photos with TRAINED PROJECTION...\n")

    for category in categories:
        folder_path = os.path.join(DEV_QUERY_FOLDER, category)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)]

        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                query_emb = get_projected_embedding(fpath)
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
    print("TRAINED PROJECTION RESULTS (development set)")
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

    print("\nCompare these numbers to your score_baseline.py results to see")
    print("whether the trained projection improved recall and/or improved")
    print("the separation between catalogue and not-in-catalogue scores.")


if __name__ == "__main__":
    main()