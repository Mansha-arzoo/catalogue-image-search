"""
split_query_photos.py
-----------------------
Splits each category's query photos into TWO groups:
    - development/   -> used repeatedly while building and tuning the app
    - held_back/      -> opened ONLY ONCE, at the very end, for the final honest score

WHY THIS MATTERS:
If you keep testing your app on the same photos you already looked at,
your results become artificially good (like practicing with the exam
paper already in hand). The held-back set stays untouched until the
very last day, so the final numbers are trustworthy.

WHAT THIS SCRIPT DOES:
- Goes through every folder inside catalogue_project/query_photos/
  (including not_in_catalogue).
- Randomly splits the images in each folder: ~60% to development,
  ~40% to held-back.
- COPIES the files (does not delete originals) into a new structure:

    catalogue_project/
        query_split/
            development/
                01_pens/
                02_brushes/
                ...
                not_in_catalogue/
            held_back/
                01_pens/
                02_brushes/
                ...
                not_in_catalogue/

HOW TO RUN (from inside catalogue_project folder):
    python split_query_photos.py

IMPORTANT:
Run this ONLY ONCE. If you run it again, it will re-randomize the
split, which defeats the purpose (you want the SAME held-back images
to stay untouched throughout the project). A fixed random seed is
used below so that even if you accidentally re-run it, you'll get the
exact same split back - but it's still best practice not to re-run it.
"""

import os
import shutil
import random

QUERY_FOLDER = "query_photos"
OUTPUT_FOLDER = "query_split"
DEV_RATIO = 0.6   # 60% development, 40% held-back

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

random.seed(42)  # fixed seed = reproducible split every time this exact script runs


def main():
    if not os.path.isdir(QUERY_FOLDER):
        print(f"Could not find '{QUERY_FOLDER}'. Run this from inside catalogue_project.")
        return

    dev_root = os.path.join(OUTPUT_FOLDER, "development")
    held_root = os.path.join(OUTPUT_FOLDER, "held_back")

    subfolders = sorted(
        f for f in os.listdir(QUERY_FOLDER)
        if os.path.isdir(os.path.join(QUERY_FOLDER, f))
    )

    summary = []

    for folder in subfolders:
        src_folder = os.path.join(QUERY_FOLDER, folder)
        files = [f for f in os.listdir(src_folder) if f.lower().endswith(VALID_EXTENSIONS)]
        random.shuffle(files)

        split_point = round(len(files) * DEV_RATIO)
        dev_files = files[:split_point]
        held_files = files[split_point:]

        dev_dest = os.path.join(dev_root, folder)
        held_dest = os.path.join(held_root, folder)
        os.makedirs(dev_dest, exist_ok=True)
        os.makedirs(held_dest, exist_ok=True)

        for f in dev_files:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(dev_dest, f))
        for f in held_files:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(held_dest, f))

        summary.append((folder, len(dev_files), len(held_files)))
        print(f"{folder}: {len(dev_files)} -> development, {len(held_files)} -> held_back")

    total_dev = sum(s[1] for s in summary)
    total_held = sum(s[2] for s in summary)
    print(f"\nTotal: {total_dev} images in development, {total_held} images in held_back.")
    print(f"\nDone. New folders created under '{OUTPUT_FOLDER}/'.")
    print("Your original query_photos folder is untouched (files were copied, not moved).")
    print("\nIMPORTANT: From now on, only use 'development' photos while building")
    print("and testing your app. Do NOT open 'held_back' until the very final")
    print("evaluation on the last day of your project.")


if __name__ == "__main__":
    main()