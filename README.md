# 😊 Smile Detection

A real-time smile detector built with Python and OpenCV. Works with a live webcam feed or static image files — no internet connection or pre-trained deep learning models required.

---

## Features

- 🎥 **Real-time webcam** detection at live frame rate
- 🖼️ **Static image** detection with annotated output saved automatically
- 💾 **Save video** output to `.avi` file
- 📊 **HUD overlay** showing face count and smiling count per frame
- 🎨 **Color-coded bounding boxes** — green for smiling, blue-orange for neutral

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.7+ |
| opencv-python | 4.x+ |

Install with:

```bash
pip install opencv-python
```

> No additional model downloads needed — Haar cascade files ship with OpenCV.

---

## Usage

### Webcam (real-time)
```bash
python smile_detection.py
```
Press **`Q`** to quit.

### Static image
```bash
python smile_detection.py --image path/to/photo.jpg
```
An annotated copy is saved as `photo_detected.jpg` in the same folder.

### Webcam + save video
```bash
python smile_detection.py --save output.avi
```

---

## How It Works

```
Frame
  │
  ▼
Convert to Grayscale + Equalize Histogram
  │
  ▼
Face Detection  ──────────────────────────────────────────┐
(haarcascade_frontalface_default.xml)                     │
  │                                                        │
  ▼                                                        │
Crop lower half of each face (mouth region ROI)           │
  │                                                        │
  ▼                                                        │
Smile Detection                                           │
(haarcascade_smile.xml)                                   │
  │                                                        │
  ▼                                                        │
Annotate frame with boxes + label ◄───────────────────────┘
  │
  ▼
Display / Save
```

1. Each frame is converted to **grayscale** and contrast-enhanced via histogram equalization.
2. **Faces** are located using a frontal face Haar cascade.
3. Only the **lower half** of each face bounding box is passed to smile detection — this focuses on the mouth area and dramatically reduces false positives.
4. If one or more smiles are found in that region, the face is labeled **Smiling 😊**; otherwise **No Smile 😐**.

---

## Tuning Parameters

Edit these values in `smile_detection.py` to adjust sensitivity:

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `minNeighbors` (smile) | `detectMultiScale` call | `22` | **Raise** to reduce false positives; **lower** to catch subtle smiles |
| `scaleFactor` (smile) | `detectMultiScale` call | `1.7` | **Lower** (e.g. `1.5`) to detect smiles at more scales |
| `minSize` (smile) | `detectMultiScale` call | `(25, 25)` | **Increase** to ignore small noise detections |
| `minNeighbors` (face) | `detectMultiScale` call | `5` | **Raise** if too many ghost faces appear |
| ROI split point | `y + h // 2` | 50% | Change to `y + h // 3` to include more of the face |

---

## Output Example

```
[RESULT] Detected 2 face(s).
  Face 1: Smiling 😊  |  bounding box: (102, 88, 210, 210)
  Face 2: Not smiling 😐  |  bounding box: (340, 70, 198, 198)

[INFO] Annotated image saved to: photo_detected.jpg
```

---

## Limitations

- Works best with **frontal, well-lit** faces.
- Haar cascades can produce **false positives** in busy backgrounds — raise `minNeighbors` if this occurs.
- Performance may degrade with **very small faces** (< 60×60 px) or extreme angles.
- For production use, consider replacing Haar cascades with a deep learning model (e.g. MediaPipe Face Mesh or a fine-tuned MobileNet).

---

## License

MIT — free to use, modify, and distribute.
