"""
vector_size_study.py
----------------------
DAY 8: Tests how shrinking the embedding vectors (using PCA) affects:
    1. QUALITY (Recall@1, Recall@5)
    2. MEMORY (how much space the gallery embeddings take up)
    3. SPEED (embedding time vs search time, measured separately)

WHAT IS PCA, IN SIMPLE WORDS:
Our embeddings currently have 2048 numbers each. Many of those numbers
carry very little useful information. PCA finds a smaller set of
numbers (e.g. 256) that keeps as much of the IMPORTANT information as
possible, while throwing away the least useful parts. Smaller vectors
take less memory and are faster to search, but may lose some accuracy.

IMPORTANT: PCA is "trained" (fitted) using ONLY the gallery images,
never the query photos - consistent with the rest of the project.

HOW TO RUN (from inside catalogue_project folder):
    python vector_size_study.py

REQUIRES:
    gallery_embeddings.npz
    query_split/development/

OUTPUT:
    Prints a results table, and saves it to vector_size_results.csv
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from sklearn.decomposition import PCA
import pandas as pd

GALLERY_EMB_FILE = "gallery_embeddings.npz"
DEV_QUERY_FOLDER = os.path.join("query_split", "development")
NOT_IN_CATALOGUE_FOLDER_NAME = "not_in_catalogue"
RESULTS_CSV = "vector_size_results.csv"

TARGET_DIMENSIONS = [2048, 512, 256, 128, 64, 32]
NUM_SEARCH_TIMING_RUNS = 50

VALID_EXTENSIONS = (
    ".jpg", ".jpeg", ".jfif", ".jpe", ".jif",
    ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif", ".avif",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_sim(query_vec, all_vecs):
    q = query_vec / np.linalg.norm(query_vec)
    a = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return a @ q


def main():
    data = np.load(GALLERY_EMB_FILE, allow_pickle=True)
    gallery_embeddings = data["embeddings"].astype(np.float32)
    gallery_labels = data["labels"]
    print(f"Loaded {gallery_embeddings.shape[0]} gallery embeddings (2048-dim).\n")

    print("Loading pretrained ResNet50 to embed development query photos...")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    base_model = models.resnet50(weights=weights)
    base_model.fc = nn.Identity()
    base_model.eval()
    base_model.to(device)
    preprocess = weights.transforms()

    def get_raw_embedding(image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = base_model(tensor)
        return emb.squeeze().cpu().numpy()

    categories = sorted(
        f for f in os.listdir(DEV_QUERY_FOLDER)
        if os.path.isdir(os.path.join(DEV_QUERY_FOLDER, f))
    )

    query_embeddings_raw = []
    query_true_labels = []
    single_query_embed_time = []

    print("Embedding all development query photos once (raw 2048-dim)...\n")
    for category in categories:
        folder_path = os.path.join(DEV_QUERY_FOLDER, category)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS)]
        for fname in files:
            fpath = os.path.join(folder_path, fname)
            try:
                t0 = time.time()
                emb = get_raw_embedding(fpath)
                single_query_embed_time.append(time.time() - t0)
            except Exception:
                continue
            query_embeddings_raw.append(emb)
            query_true_labels.append(category)
        print(f"  Done: {category}")

    query_embeddings_raw = np.array(query_embeddings_raw, dtype=np.float32)
    query_true_labels = np.array(query_true_labels)
    avg_embed_time_ms = np.mean(single_query_embed_time) * 1000

    print(f"\nTotal development queries embedded: {len(query_embeddings_raw)}")
    print(f"Average embedding time per photo: {avg_embed_time_ms:.1f} ms")
    print("(This embedding time is the SAME at every vector size below,")
    print(" since PCA is applied AFTER the ResNet50 embedding step.)\n")

    results = []

    for dim in TARGET_DIMENSIONS:
        print(f"--- Testing dimension: {dim} ---")

        if dim == 2048:
            gallery_reduced = gallery_embeddings
            query_reduced = query_embeddings_raw
        else:
            pca = PCA(n_components=dim, random_state=42)
            pca.fit(gallery_embeddings)
            gallery_reduced = pca.transform(gallery_embeddings).astype(np.float32)
            query_reduced = pca.transform(query_embeddings_raw).astype(np.float32)

        top1_correct = 0
        top5_correct = 0
        total_catalogue = 0

        for i, true_label in enumerate(query_true_labels):
            if true_label == NOT_IN_CATALOGUE_FOLDER_NAME:
                continue
            sims = cosine_sim(query_reduced[i], gallery_reduced)
            top5_idx = np.argsort(sims)[::-1][:5]
            top5_labels = gallery_labels[top5_idx]

            total_catalogue += 1
            if top5_labels[0] == true_label:
                top1_correct += 1
            if true_label in top5_labels:
                top5_correct += 1

        recall1 = top1_correct / total_catalogue
        recall5 = top5_correct / total_catalogue

        memory_bytes = gallery_reduced.nbytes
        memory_kb = memory_bytes / 1024
        memory_per_image_bytes = memory_bytes / gallery_reduced.shape[0]

        search_times = []
        for _ in range(NUM_SEARCH_TIMING_RUNS):
            sample_query = query_reduced[np.random.randint(0, len(query_reduced))]
            t0 = time.time()
            _ = cosine_sim(sample_query, gallery_reduced)
            search_times.append(time.time() - t0)
        avg_search_time_ms = np.mean(search_times) * 1000

        results.append({
            "vector_size": dim,
            "recall@1": round(recall1, 3),
            "recall@5": round(recall5, 3),
            "memory_KB_total": round(memory_kb, 1),
            "memory_bytes_per_image": round(memory_per_image_bytes, 1),
            "embedding_time_ms": round(avg_embed_time_ms, 2),
            "search_time_ms": round(avg_search_time_ms, 3),
            "total_time_ms": round(avg_embed_time_ms + avg_search_time_ms, 2),
        })

        print(f"  Recall@1={recall1:.1%}  Recall@5={recall5:.1%}  "
              f"Memory={memory_kb:.1f} KB  Search time={avg_search_time_ms:.3f} ms\n")

    df = pd.DataFrame(results)
    print("=" * 70)
    print("VECTOR SIZE STUDY - FINAL RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved results table to '{RESULTS_CSV}'.")
    print("Use this table directly in your report's 'Size against quality table'.")


if __name__ == "__main__":
    main()