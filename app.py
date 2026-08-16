"""
app.py
-------
DAY 5 (IMPROVED): The browser application. Lets someone upload a phone
photo (or take one with their camera) and see the closest catalogue
items with similarity scores, or a "no match" message if the photo
doesn't look like anything in the catalogue.

Uses the BASELINE model (plain pretrained ResNet50), since it
outperformed the trained projection on the development set.

NEW IN THIS VERSION:
    - Sidebar with catalogue stats (categories, total gallery images)
    - Adjustable threshold slider (great for live demos - shows how
      the cut-off trade-off works in real time)
    - Bar chart of the top-5 similarity scores
    - Camera input option, not just file upload
    - Cleaner layout/styling

HOW TO RUN (from inside catalogue_project folder):
    streamlit run app.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import streamlit as st
import pandas as pd

GALLERY_EMB_FILE = "gallery_embeddings.npz"
GALLERY_FOLDER = "gallery"
DEFAULT_THRESHOLD = 0.45
TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    preprocess = weights.transforms()
    return model, preprocess


@st.cache_resource
def load_gallery():
    data = np.load(GALLERY_EMB_FILE, allow_pickle=True)
    return data["embeddings"], data["labels"], data["filenames"]


def get_embedding(model, preprocess, pil_image):
    tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.squeeze().cpu().numpy()


def cosine_sim(query_vec, all_vecs):
    q = query_vec / np.linalg.norm(query_vec)
    a = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    return a @ q


def find_example_image(category, filename):
    path = os.path.join(GALLERY_FOLDER, category, filename)
    return path if os.path.exists(path) else None


def pretty_name(category):
    """Turns '01_pens' into 'Pens' for nicer display."""
    parts = category.split("_", 1)
    name = parts[1] if len(parts) > 1 else category
    return name.replace("_", " ").title()


st.set_page_config(page_title="Catalogue Image Search", page_icon="🔍", layout="centered")

model, preprocess = load_model()
gallery_embeddings, gallery_labels, gallery_filenames = load_gallery()
categories = sorted(set(gallery_labels))

with st.sidebar:
    st.header("About this catalogue")
    st.metric("Categories", len(categories))
    st.metric("Gallery images", len(gallery_labels))
    st.divider()
    threshold = st.slider(
        "Similarity cut-off",
        min_value=0.20,
        max_value=0.70,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Results below this similarity score are refused as 'no match'. "
             "Lower = more lenient (more false matches). Higher = stricter "
             "(more valid items wrongly refused).",
    )
    st.caption(f"Default cut-off from Day 7 evaluation: {DEFAULT_THRESHOLD}")
    st.divider()
    with st.expander("Browse catalogue categories"):
        for c in categories:
            st.write(f"- {pretty_name(c)}")

st.title("🔍 Catalogue Image Search")
st.write(
    "Upload a photo, or take one with your camera. The app compares it "
    "against the catalogue and shows the closest matches, or tells you "
    "if the item isn't in the catalogue."
)
st.info(
    "Your photo is kept only in memory for this request and is not stored anywhere.",
    icon="ℹ️",
)

input_mode = st.radio("Input method:", ["Upload a photo", "Use camera"], horizontal=True)

uploaded_file = None
if input_mode == "Upload a photo":
    uploaded_file = st.file_uploader(
        "Choose a photo...", type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    try:
        start_time = time.time()

        image = Image.open(uploaded_file)
        st.image(image, caption="Your photo", width=300)

        embed_start = time.time()
        query_embedding = get_embedding(model, preprocess, image)
        embed_time = time.time() - embed_start

        search_start = time.time()
        sims = cosine_sim(query_embedding, gallery_embeddings)
        top_idx = np.argsort(sims)[::-1][:TOP_K]
        search_time = time.time() - search_start

        total_time = time.time() - start_time
        best_score = sims[top_idx[0]]

        st.divider()

        if best_score < threshold:
            st.warning(
                f"**No match found.** This doesn't look like anything in "
                f"the catalogue (best similarity score: {best_score:.2f}, "
                f"below the {threshold:.2f} cut-off)."
            )
        else:
            st.subheader("Closest matches")

            chart_data = pd.DataFrame({
                "category": [pretty_name(gallery_labels[i]) for i in top_idx],
                "similarity": [float(sims[i]) for i in top_idx],
            }).set_index("category")
            st.bar_chart(chart_data)

            for rank, idx in enumerate(top_idx, start=1):
                category = gallery_labels[idx]
                score = sims[idx]
                example_path = find_example_image(category, gallery_filenames[idx])

                cols = st.columns([1, 3])
                with cols[0]:
                    if example_path:
                        st.image(example_path, width=100)
                with cols[1]:
                    st.write(f"**{rank}. {pretty_name(category)}**")
                    st.progress(min(float(score), 1.0))
                    st.caption(f"similarity score: {score:.3f}")

        with st.expander("Timing details"):
            st.write(f"Embedding time: {embed_time*1000:.0f} ms")
            st.write(f"Search time: {search_time*1000:.0f} ms")
            st.write(f"Total time: {total_time*1000:.0f} ms")

    except Exception as e:
        st.error(
            "Sorry, something went wrong processing this file. "
            "Please make sure you uploaded a valid image."
        )
        st.caption(f"(Technical detail: {e})")