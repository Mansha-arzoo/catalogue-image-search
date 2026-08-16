# Run Log — Catalogue Image Search Project
Mansha Arzo | Centre for AI & BigData, Namal University Mianwali

One line per run: what changed, and what happened.

---

1. **create_folders.py** — Created initial gallery folder structure (30 categories).
   Result: 30 empty folders created successfully.

2. **download_images.py** (icrawler, Bing) — Attempted automatic image download for gallery categories.
   Result: Many irrelevant/low-quality results; switched to manual, curated image collection instead.

3. **Manual gallery collection** — Finalized 26 categories (after several revisions with supervisor's input), collected images manually (mix of own phone photos + curated internet images per supervisor's guidance).
   Result: 819 gallery images across 26 categories, 25–41 images per category.

4. **check_dataset.py** — Ran blur/duplicate/corrupt-file check on gallery images.
   Result: A few exact duplicates removed; most "blurry" flags were false positives on plain-background product photos (kept as-is).

5. **Query photo collection** — Took messy/casual phone photos of 26 available items, from internet, added 23 "not_in_catalogue" photos.
   Result: ~230 query photos collected across 26 categories + not_in_catalogue.

6. **check_query_dataset.py** — Quality check on query photos.
   Result: Mostly normal blur readings for real phone photos (WhatsApp compression); no major issues.

7. **split_query_photos.py** — Split query photos into development (60%) / held-back (40%) sets, fixed random seed.
   Result: Held-back set created and set aside untouched for final evaluation.

8. **extract_embeddings.py** (attempt 1) — Ran embedding extraction on gallery images.
   Result: Failed — wrong folder path (GALLERY_FOLDER pointed one level too high). Fixed path.

9. **extract_embeddings.py** (attempt 2) — Re-ran after path fix.
   Result: Only 526/819 images processed — many `.jfif` files and corrupted "Chrome HTML Document" files were silently skipped.

10. **extract_embeddings.py** (attempt 3) — Expanded VALID_EXTENSIONS to cover all common image formats (.jfif, .heic, .avif, etc.) and installed pillow-heif/pillow-avif-plugin.
    Result: All 819 gallery images successfully embedded (2048-dim each). Saved to gallery_embeddings.npz.

11. **test_embeddings.py** — Quick sanity check: nearest-neighbour search within gallery only.
    Result: 8/8 random test images correctly matched to their own category — embeddings behaving as expected.

12. **score_baseline.py** — Evaluated baseline (plain pretrained ResNet50) on development query photos.
    Result: Recall@1 = 85.6%, Recall@5 = 95.2%. Catalogue avg score 0.522, not-in-catalogue avg 0.400.

13. **train_projection.py** (attempt 1, 40 epochs, no regularization) — Trained a projection head on gallery embeddings.
    Result: Reached 100% training accuracy by epoch 10 — clear sign of overfitting.

14. **score_projection.py** (attempt 1) — Evaluated the overfit projection on development query photos.
    Result: Recall@1 = 78.4%, Recall@5 = 83.2% — worse than baseline, confirming overfitting hurt generalization.

15. **train_projection.py** (attempt 2, 15 epochs, dropout 0.3, weight decay, validation split) — Retrained with regularization.
    Result: Validation accuracy stabilized at 95.9%, no runaway overfitting.

16. **score_projection.py** (attempt 2) — Re-evaluated the regularized projection.
    Result: Recall@1 = 84.0%, Recall@5 = 88.0% — improved vs attempt 1, but still below baseline. Decision: use baseline model going forward, report both results with explanation (small dataset + already-distinct categories favour the baseline).

17. **choose_cutoff.py** — Swept thresholds 0.20–0.70 on development set scores to find a cut-off.
    Result: Recommended cut-off = 0.45 (equal error rate point): 24.0% false refusals, 21.4% false matches on this small development sample. Plot saved as cutoff_curve.png.

18. **app.py** (version 1) — Built basic Streamlit app: upload photo, show top-5 matches, refuse below threshold.
    Result: Working locally, handles bad uploads gracefully.

19. **app.py** (version 2) — Added sidebar stats, adjustable threshold slider, bar chart of scores, camera input, improved styling.
    Result: Working locally, more demo-friendly.

20. **vector_size_study.py** — Tested PCA-reduced embeddings at 2048/512/256/128/64/32 dimensions on development queries.
    Result: 128-dim gave near-identical accuracy (Recall@1 84.0%, Recall@5 95.2%) to the full 2048-dim, at ~16x less memory and ~35x faster search. Chose 128-dim as the best trade-off. Results saved to vector_size_results.csv.

21. **failure_analysis.py** — Identified all development-set queries where the top-1 match was wrong.
    Result: 18 failure cases found and saved to failure_cases/ with failure_log.csv. Manual review found 3 main causes: dominant background patterns, inherently similar packaging categories, and a few genuinely close categories (e.g. bottles vs shampoo bottles).

22. **GitHub setup** — Initialized git repo, pushed full project (gallery, query photos, all scripts) to GitHub.
    Result: Successfully pushed (291 MB). Repository: github.com/Mansha-arzoo/catalogue-image-search

23. **Deployment attempt 1 (Streamlit Community Cloud)** — Deployed app.py directly.
    Result: Failed — ModuleNotFoundError on import torch. Default requirements.txt pulled the full GPU-enabled torch build, too large for the free tier.

24. **Deployment attempt 2** — Updated requirements.txt to install CPU-only torch via --extra-index-url https://download.pytorch.org/whl/cpu, pushed fix, rebooted app.
    Result: Deployment successful. App live and working on Streamlit Community Cloud.

---

(To be continued: idle/warm timing measurements, held-back evaluation, final report.)
