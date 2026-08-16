"""
train_projection.py
---------------------
DAY 6: Trains a small "projection" layer on top of the frozen pretrained
ResNet50 embeddings, using ONLY the gallery images (query photos are never
used for training - they stay purely for testing).

WHAT THIS DOES, IN SIMPLE WORDS:
- Takes the 2048-number embeddings we already extracted for the gallery.
- Trains a small extra neural network (a "projection head") that learns
  to tell your 26 categories apart.
- The side-effect of learning this is that the projection's OUTPUT
  vectors end up clustering same-category images closer together, and
  different categories further apart - exactly what we want for search.
- We THROW AWAY the classification part after training, and keep only
  the projection layer, which we then use to re-embed everything.

OUTPUT FILES:
    projection_weights.pth        - the trained projection layer
    gallery_embeddings_projected.npz - gallery images re-embedded with
                                        the trained projection (256 numbers each)

HOW TO RUN (from inside catalogue_project folder):
    python train_projection.py

REQUIRES:
    gallery_embeddings.npz (already created in the baseline step)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

GALLERY_EMB_FILE = "gallery_embeddings.npz"
PROJECTION_OUTPUT_FILE = "projection_weights.pth"
PROJECTED_EMB_OUTPUT_FILE = "gallery_embeddings_projected.npz"

INPUT_DIM = 2048          # size of the raw ResNet50 embedding
PROJECTED_DIM = 256       # size of our new, trained embedding
HIDDEN_DIM = 512
EPOCHS = 15                # reduced from 40 - fewer epochs to avoid overfitting
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4         # regularization - discourages the model from memorizing
DROPOUT = 0.3                # randomly "turns off" neurons during training so the
                              # model can't rely too heavily on any single feature
VAL_SPLIT = 0.15            # held-out slice of GALLERY (not query photos) to watch for overfitting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProjectionHead(nn.Module):
    """The small trainable network: 2048 -> 512 -> 256 (this part we KEEP)."""
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


def main():
    # ---------------------------------------------------------
    # STEP 1: Load the gallery embeddings + category labels
    # ---------------------------------------------------------
    data = np.load(GALLERY_EMB_FILE, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"]

    unique_categories = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_categories)}
    label_indices = np.array([label_to_idx[l] for l in labels])
    num_classes = len(unique_categories)

    print(f"Loaded {embeddings.shape[0]} gallery embeddings across {num_classes} categories.\n")

    X = torch.tensor(embeddings)
    y = torch.tensor(label_indices, dtype=torch.long)

    # Split off a small validation slice from the GALLERY (still not query
    # photos) so we can watch for overfitting during training.
    num_samples = len(X)
    indices = torch.randperm(num_samples)
    val_size = int(num_samples * VAL_SPLIT)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------------
    # STEP 2: Build the projection head + a temporary classifier on top
    # (the classifier is only used to guide training - we discard it after)
    # ---------------------------------------------------------
    projection = ProjectionHead(INPUT_DIM, HIDDEN_DIM, PROJECTED_DIM, dropout=DROPOUT).to(device)
    classifier = nn.Linear(PROJECTED_DIM, num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(projection.parameters()) + list(classifier.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # STEP 3: Train, checking validation accuracy every epoch
    # ---------------------------------------------------------
    print("Training projection layer...\n")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        projection.train()
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            projected = projection(batch_x)
            logits = classifier(projected)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation check (no gradient updates, dropout turned off)
        projection.eval()
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(projection(X_val.to(device)))
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = (val_preds == y_val).float().mean().item()

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.1%}  val_acc={val_acc:.1%}")

        # Keep the version of the model that did best on validation data
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in projection.state_dict().items()}

    # Restore the best-performing version (helps avoid using an overfit epoch)
    projection.load_state_dict(best_state)
    print(f"\nUsing the epoch with the best validation accuracy: {best_val_acc:.1%}")

    # ---------------------------------------------------------
    # STEP 4: Save the trained projection layer (NOT the classifier)
    # ---------------------------------------------------------
    torch.save(projection.state_dict(), PROJECTION_OUTPUT_FILE)
    print(f"\nSaved trained projection layer to '{PROJECTION_OUTPUT_FILE}'.")

    # ---------------------------------------------------------
    # STEP 5: Re-embed every gallery image using the trained projection
    # ---------------------------------------------------------
    projection.eval()
    with torch.no_grad():
        projected_embeddings = projection(X.to(device)).cpu().numpy()

    np.savez(
        PROJECTED_EMB_OUTPUT_FILE,
        embeddings=projected_embeddings,
        labels=labels,
        filenames=data["filenames"],
    )
    print(f"Saved projected gallery embeddings to '{PROJECTED_EMB_OUTPUT_FILE}'.")
    print(f"Each embedding now has {projected_embeddings.shape[1]} numbers (was 2048).")


if __name__ == "__main__":
    main()