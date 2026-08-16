"""
check_query_dataset.py
------------------------
Checks the quality of the QUERY PHOTOS (messy phone photos + not-in-catalogue
photos) before we split them into development and held-back sets.

Checks for each folder inside catalogue_project/query_photos/:
1. How many images are in each folder.
2. Blurry images.
3. Files that are not actually valid images (e.g. accidentally saved
   HTML pages instead of the image itself - this happened before with
   the gallery photos, so worth checking again here).

Does NOT delete anything automatically - only reports, so you can decide.

BEFORE RUNNING (already installed if you did the earlier gallery check):
    pip install opencv-python-headless imagehash pillow

HOW TO RUN (from inside catalogue_project folder):
    python check_query_dataset.py

OUTPUT:
    query_dataset_report.txt - open with Notepad to read the results.
"""

import os
import cv2

QUERY_FOLDER = "query_photos"
REPORT_FILE = "query_dataset_report.txt"

BLUR_THRESHOLD = 100.0

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


def is_blurry(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD, variance


def check_folder(folder_path, report_lines):
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(VALID_EXTENSIONS)]
    non_image_files = [f for f in all_files if not f.lower().endswith(VALID_EXTENSIONS)
                        and os.path.isfile(os.path.join(folder_path, f))]

    report_lines.append(f"\n=== {os.path.basename(folder_path)} ({len(image_files)} images) ===")

    if non_image_files:
        report_lines.append(f"  NON-IMAGE FILES FOUND (check these - may be broken downloads):")
        for f in non_image_files:
            report_lines.append(f"    - {f}")

    if len(image_files) < 6:
        report_lines.append(f"  WARNING: Only {len(image_files)} images - aim for at least 6-8.")

    for fname in image_files:
        fpath = os.path.join(folder_path, fname)
        result = is_blurry(fpath)
        if result is None:
            report_lines.append(f"  CORRUPTED/UNREADABLE: {fname}")
            continue
        blurry, variance = result
        if blurry:
            report_lines.append(f"  BLURRY (score={variance:.1f}, lower=more blurry): {fname}")


def main():
    if not os.path.isdir(QUERY_FOLDER):
        print(f"Could not find '{QUERY_FOLDER}'. Run this script from inside catalogue_project.")
        return

    report_lines = ["QUERY PHOTOS QUALITY REPORT", "=" * 40]

    subfolders = sorted(
        f for f in os.listdir(QUERY_FOLDER)
        if os.path.isdir(os.path.join(QUERY_FOLDER, f))
    )

    total_images = 0
    for folder in subfolders:
        check_folder(os.path.join(QUERY_FOLDER, folder), report_lines)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Done! Report saved to '{REPORT_FILE}'. Open it with Notepad.")


if __name__ == "__main__":
    main()