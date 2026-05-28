
import cv2
import argparse
import sys


def load_cascades():
    """Load Haar cascade classifiers for face and smile detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    if face_cascade.empty() or smile_cascade.empty():
        print("[ERROR] Could not load Haar cascade files. Make sure opencv-python is installed correctly.")
        sys.exit(1)

    return face_cascade, smile_cascade


def detect_smile_in_frame(frame, face_cascade, smile_cascade):
    """
    Detect faces and smiles in a single frame.

    Returns:
        annotated frame with bounding boxes drawn
        list of (face_rect, smiling: bool) tuples
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for better detection

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results = []

    for (x, y, w, h) in faces:
        # Region of interest — focus smile detection on the lower half of the face
        roi_gray  = gray[y + h // 2 : y + h, x : x + w]
        roi_color = frame[y + h // 2 : y + h, x : x + w]

        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,   # Higher value = fewer false positives
            minSize=(25, 25),
        )

        is_smiling = len(smiles) > 0
        results.append(((x, y, w, h), is_smiling))

        # Draw face rectangle
        face_color = (0, 255, 0) if is_smiling else (255, 100, 50)
        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)

        # Label above the face box
        label = "😊 Smiling!" if is_smiling else "😐 No Smile"
        cv2.putText(
            frame, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            face_color, 2,
        )

        # Draw smile rectangles (offset to full frame coordinates)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(
                roi_color,
                (sx, sy), (sx + sw, sy + sh),
                (0, 200, 255), 2,
            )

    return frame, results


def run_webcam(save_path=None):
    """Run real-time smile detection from the default webcam."""
    face_cascade, smile_cascade = load_cascades()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        print(f"[INFO] Saving output to: {save_path}")

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame.")
            break

        frame, results = detect_smile_in_frame(frame, face_cascade, smile_cascade)

        # HUD overlay
        face_count    = len(results)
        smiling_count = sum(1 for _, s in results if s)
        hud = f"Faces: {face_count}  |  Smiling: {smiling_count}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Smile Detection  —  press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


def run_image(image_path):
    """Run smile detection on a static image and display the result."""
    face_cascade, smile_cascade = load_cascades()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    frame, results = detect_smile_in_frame(frame, face_cascade, smile_cascade)

    print(f"\n[RESULT] Detected {len(results)} face(s).")
    for i, (rect, smiling) in enumerate(results, 1):
        status = "Smiling 😊" if smiling else "Not smiling 😐"
        print(f"  Face {i}: {status}  |  bounding box: {rect}")

    # Save annotated image next to the original
    out_path = image_path.rsplit(".", 1)[0] + "_detected.jpg"
    cv2.imwrite(out_path, frame)
    print(f"\n[INFO] Annotated image saved to: {out_path}")

    cv2.imshow("Smile Detection  —  press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Smile Detection with OpenCV")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file (omit for webcam mode)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save webcam output to this .avi file path")
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_webcam(save_path=args.save)


if __name__ == "__main__":
    main()
