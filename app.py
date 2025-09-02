from ultralytics import YOLO
import cv2
from collections import deque
import os
from datetime import datetime

# ===== CONFIGURATION =====
MODEL_PATH = "best.pt"   # Palitan kung iba ang filename ng model mo
CAMERA_INDEX = 0
PRE_SECONDS = 5
POST_SECONDS = 5
MAX_ALLOWED_RIDERS = 2   # Max riders sa motor

# Class indexes sa model.names (check mo output ng print sa baba)
CLASS_MOTORCYCLE = 0
CLASS_NO_HELMET = 1
CLASS_PROPER_HELMET = 2
CLASS_WRONG_HELMET = 3

# ===== LOAD MODEL =====
model = YOLO("best.pt")
print("[INFO] Model classes:", model.names)

# ===== VIDEO CAPTURE =====
cap = cv2.VideoCapture(CAMERA_INDEX)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
buffer = deque(maxlen=int(PRE_SECONDS * fps))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save sa buffer para pre-violation recording
    buffer.append(frame.copy())

    # Run YOLO detection
    results = model(frame, imgsz=640, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    # Variables
    violation_detected = False
    violation_type = ""
    person_count = 0

    # Check detections
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])

            # Count tao sa motor
            if cls_id in [CLASS_NO_HELMET, CLASS_PROPER_HELMET, CLASS_WRONG_HELMET]:
                person_count += 1

            # Helmet violations
            if cls_id in [CLASS_NO_HELMET, CLASS_WRONG_HELMET]:
                violation_detected = True
                violation_type = model.names[cls_id]

    # Overloading violation
    if person_count > MAX_ALLOWED_RIDERS:
        violation_detected = True
        violation_type = f"overloading_{person_count}_persons"

    # Kung may violation, save clip
    if violation_detected:
        cv2.putText(annotated_frame, f"VIOLATION: {violation_type.upper()}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)

        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = f"violations/{today}"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{violation_type}_{datetime.now().strftime('%H-%M-%S')}.avi")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, fps, (frame.shape[1], frame.shape[0]))

        # Pre-violation frames
        for f in buffer:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(f, ts, (10, f.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(f)

        # Post-violation frames
        for _ in range(int(fps * POST_SECONDS)):
            ret, post_frame = cap.read()
            if not ret:
                break
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(post_frame, ts, (10, post_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(post_frame)
            cv2.imshow("Helmet & Overloading Detection", post_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        buffer.clear()
        print(f"[SAVED] {violation_type.upper()} clip: {filename}")

    # Show live detection feed
    cv2.imshow("Helmet & Overloading Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)
model = YOLO('c:/Users/admin/Desktop/Thesis 2/best.pt')
violations = []

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # Detect violation (example: any detection is a violation)
        violation_text = ""
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                violation_text = model.names[cls_id]
                violations.append(violation_text)

        # Encode frame for HTML
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/violations')
def get_violations():
    # Only show last 10 violations
    return jsonify(violations[-10:])

if __name__ == '__main__':
    app.run(debug=True)

import webbrowser
if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True)