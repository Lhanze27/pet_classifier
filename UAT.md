# User Acceptance Test — Pet Breed Classifier

**System:** Pet Breed Classifier (EfficientNetV2B0, 37 Oxford-IIIT breeds)
**Components under test:** `04_server.py` (FastAPI) + `web/` (editorial frontend)
**Tester:** ____________________   **Date:** ____________________
**Build / model file:** `outputs/models/final_model.keras` (size: ______, trained: ______)

---

## 1. Pre-conditions

| # | Check | Pass |
|---|-------|------|
| P1 | Python 3.10 or 3.11 installed | ☐ |
| P2 | Virtual environment activated (`(venv)` visible in prompt) | ☐ |
| P3 | Dependencies installed (`pip install -r requirements.txt` completed) | ☐ |
| P4 | Trained model exists at `outputs/models/final_model.keras` | ☐ |
| P5 | `outputs/class_names.txt` exists and lists 37 classes | ☐ |
| P6 | Modern browser available (Chrome/Edge/Firefox latest) | ☐ |
| P7 | Webcam connected (for live-camera tests) | ☐ |

---

## 2. Server startup

| TC | Steps | Expected | Actual | Pass |
|----|-------|----------|--------|------|
| TC-01 | From `pet_classifier/`, run `python 04_server.py` | Console prints `Loading model...` then `Loaded 37 classes (12 cats, 25 dogs)` and uvicorn listens on `http://127.0.0.1:8000` | | ☐ |
| TC-02 | Delete/rename `final_model.keras`, run server | Startup fails with `FileNotFoundError: ... Run 01_train_model.py first.` | | ☐ |
| TC-03 | Open `http://127.0.0.1:8000` in browser | Editorial homepage loads — Fraunces headline, paper-grain background, two tabs visible (Upload & Paste, Live Camera) | | ☐ |
| TC-04 | View page source / network tab | `style.css` and `app.js` load with HTTP 200 from `/static/` | | ☐ |

---

## 3. API — `/api/breeds`

| TC | Steps | Expected | Pass |
|----|-------|----------|------|
| TC-05 | GET `http://127.0.0.1:8000/api/breeds` | JSON with keys `cats`, `dogs`, `total`, `threshold`. `total == 37`, `threshold == 0.5` | ☐ |
| TC-06 | Inspect `cats` list | Each entry has `name` (snake_case) and `pretty` (Title Case). 12 entries. | ☐ |
| TC-07 | Inspect `dogs` list | 25 entries. Lowercase `name` field. | ☐ |

---

## 4. API — `/api/predict`

| TC | Steps | Expected | Pass |
|----|-------|----------|------|
| TC-08 | POST `/api/predict` with valid base64 JPEG of a cat | 200 OK, `top` array of 5 predictions sorted desc by `prob`, `species == "cat"` for top-1 if cat is clearly visible | ☐ |
| TC-09 | POST with `data:image/png;base64,...` data URL | Accepted (prefix stripped server-side), valid response | ☐ |
| TC-10 | POST with invalid base64 string | 400 Bad Request, detail `"Invalid base64 image"` | ☐ |
| TC-11 | POST with valid base64 but non-image bytes | 400 Bad Request, detail `"Could not decode image"` | ☐ |
| TC-12 | POST clear image of a known breed | Top-1 `prob >= 0.50`, `low_confidence == false` | ☐ |
| TC-13 | POST a blurry / non-pet image | `low_confidence == true` | ☐ |
| TC-14 | Sum of all 37 class probs (call with `probs` exposed via test) | Approx. 1.0 (softmax) | ☐ |

---

## 5. Upload & Paste tab

| TC | Steps | Expected | Pass |
|----|-------|----------|------|
| TC-15 | Drag a JPG of a Bengal cat onto the drop zone | Image preview appears in the panel | ☐ |
| TC-16 | Click "Classify specimen" | Top-5 results render with animated probability bars; top-1 breed name pretty-printed; species label shown | ☐ |
| TC-17 | Click the file-browse control, select a PNG | Image loads identically to drag/drop | ☐ |
| TC-18 | Copy an image to clipboard, paste (Ctrl+V) on the page | Image appears in preview | ☐ |
| TC-19 | Classify a non-pet image (e.g. landscape) | Low-confidence warning displayed | ☐ |
| TC-20 | Classify, then drop a new image and classify again | Previous result replaced, no stale UI | ☐ |
| TC-21 | Try to classify with no image loaded | Button disabled OR clear error message shown | ☐ |

---

## 6. Live Camera tab

| TC | Steps | Expected | Pass |
|----|-------|----------|------|
| TC-22 | Switch to "Live Camera" tab first time | Browser prompts for camera permission | ☐ |
| TC-23 | Grant permission | Webcam feed appears, mirrored horizontally for the viewer | ☐ |
| TC-24 | Hold a pet (or pet photo) in front of camera | Predictions update ~3–4× per second (interval 280 ms) | ☐ |
| TC-25 | Deny camera permission | Friendly error message shown, no JS console crash | ☐ |
| TC-26 | Switch back to Upload tab while camera running | Camera stream stops or pauses cleanly (no zombie capture) | ☐ |
| TC-27 | Compare server-received frame orientation | Model receives natural (non-mirrored) orientation — verify dog/cat held to the user's right is classified consistently | ☐ |

---

## 7. Result correctness (smoke spot-checks)

Use known-good reference images (one per row).

| TC | Image | Expected top-1 contains | Pass |
|----|-------|------------------------|------|
| TC-28 | Pug photo | `pug` | ☐ |
| TC-29 | Siamese cat photo | `Siamese` | ☐ |
| TC-30 | Beagle photo | `beagle` | ☐ |
| TC-31 | Maine Coon photo | `Maine_Coon` | ☐ |
| TC-32 | Image with two pets | Top-5 includes both species reasonably | ☐ |

---

## 8. Non-functional

| TC | Check | Target | Pass |
|----|-------|--------|------|
| TC-33 | Cold prediction latency (first request after startup) | < 5 s | ☐ |
| TC-34 | Warm prediction latency | < 500 ms on GPU, < 2 s on CPU | ☐ |
| TC-35 | Server stays up after 50 consecutive predictions | No crashes, no memory growth > 500 MB | ☐ |
| TC-36 | Page renders correctly at 1920×1080 and 1366×768 | No overflow, no broken grid | ☐ |
| TC-37 | Fonts (Fraunces, Newsreader, JetBrains Mono) load | No FOUT longer than 1 s | ☐ |
| TC-38 | Stop server with Ctrl+C | Graceful shutdown, no traceback | ☐ |

---

## 9. Sign-off

| Role | Name | Result (Pass/Fail) | Date | Signature |
|------|------|--------------------|------|-----------|
| Tester | | | | |
| Developer | | | | |
| Project Adviser | | | | |

**Defects raised:** ____________________________________________________________

**Overall verdict:** ☐ Accepted   ☐ Accepted with minor issues   ☐ Rejected
