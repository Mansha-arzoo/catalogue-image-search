"""
test_embeddings.py
--------------------
QUICK SANITY CHECK: This does NOT use query photos. It simply checks
whether the embeddings we created make sense, by taking a few random
gallery images and asking: "which OTHER gallery images are closest to
this one?"

If the embeddings are working correctly, the closest matches to (say)
a pen image should mostly be OTHER pen images - not cups or shoes.

HOW TO RUN:
    python test_embeddings.py

WHAT TO LOOK FOR IN THE OUTPUT:
    For each randomly picked image, you'll see its TRUE category and
    the top 5 closest matches found (with their categories and
    similarity scores). If most of the top 5 matches share the same
    category as the picked image, the embeddings are working well.
"""

import numpy as np

EMBEDDINGS_FILE = "gallery_embeddings.npz"
NUM_TEST_SAMPLES = 8   # how many random images to test
TOP_K = 5               # how many closest matches to show per test image


def cosine_similarity_matrix(query_vec, all_vecs):
    """Computes cosine similarity between one vector and many vectors."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return all_norms @ query_norm


def main():
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    filenames = data["filenames"]

    print(f"Loaded {embeddings.shape[0]} embeddings across {len(set(labels))} categories.\n")

    rng = np.random.default_rng(seed=42)
    test_indices = rng.choice(len(embeddings), size=NUM_TEST_SAMPLES, replace=False)

    correct_top1 = 0

    for idx in test_indices:
        query_vec = embeddings[idx]
        true_label = labels[idx]
        true_fname = filenames[idx]

        sims = cosine_similarity_matrix(query_vec, embeddings)
        sims[idx] = -1  # exclude comparing the image to itself

        top_k_idx = np.argsort(sims)[::-1][:TOP_K]

        print(f"Query image: {true_fname}  (true category: {true_label})")
        for rank, match_idx in enumerate(top_k_idx, start=1):
            match_label = labels[match_idx]
            match_score = sims[match_idx]
            flag = "OK" if match_label == true_label else "  "
            print(f"   {rank}. [{flag}] {match_label:25s} score={match_score:.3f}  ({filenames[match_idx]})")

        if labels[top_k_idx[0]] == true_label:
            correct_top1 += 1
        print()

    print("=" * 50)
    print(f"Top-1 accuracy on this random sample: {correct_top1}/{NUM_TEST_SAMPLES}")
    print("(This is just a rough sanity check, not the real evaluation.")
    print(" The real evaluation happens later using your query photos.)")


if __name__ == "__main__":
    main()