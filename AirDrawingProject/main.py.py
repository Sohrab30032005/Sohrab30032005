import cv2
import mediapipe as mp
import numpy as np
import math
import os

# ================= CONFIG =================
REAL_IMG_DIR = "real_images"
prediction = "None"

subjects = {
    1: "Social",
    2: "Science",
    3: "Maths",
    4: "General"
}
current_subject = "General"

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

canvas = None
mask = None
prev_x, prev_y = None, None
alpha = 0.6  # smoothing

print("Draw in air | 1-4 select subject | P = predict | C = clear | Q = quit")

# ================= IMAGE LOADER =================
def load_real_image(name):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(REAL_IMG_DIR, name + ext)
        if os.path.exists(path):
            return cv2.imread(path)
    return None

# ================= SHAPE → IMAGE MAPPING =================
def map_shape_to_image(shape, subject):
    # Circle has subject-specific mappings
    if shape == "Circle":
        if subject == "Social":
            return "globe"
        elif subject == "Science":
            return "cell"
        elif subject == "Maths":
            return "circle"
        else:
            return "circle"

    # Square mappings for all subjects
    if shape == "Square":
        if subject == "Social":
            return "building"      # e.g., school, courthouse
        elif subject == "Science":
            return "microchip"     # square chip
        elif subject == "Maths":
            return "square_grid"   # graph paper/grid
        else:
            return "square"

    # Triangle mappings for all subjects
    if shape == "Triangle":
        if subject == "Social":
            return "pyramid"       # ancient structures
        elif subject == "Science":
            return "flask"         # conical flask outline
        elif subject == "Maths":
            return "set_square"    # geometry tool
        else:
            return "triangle"

    # Rectangle mappings for all subjects
    if shape == "Rectangle":
        if subject == "Social":
            return "book"          # book/door
        elif subject == "Science":
            return "battery"       # AA battery shape
        elif subject == "Maths":
            return "ruler"         # measuring ruler
        else:
            return "rectangle"

    # Ellipse mappings for all subjects
    if shape == "Ellipse":
        if subject == "Social":
            return "football"      # rugby/american football
        elif subject == "Science":
            return "orbit"         # planetary orbit
        elif subject == "Maths":
            return "ellipse_curve" # mathematical ellipse
        else:
            return "ellipse"

    # Line mapping (only in General)
    if shape == "Line":
        return "line"

    return None

# ================= SHAPE DETECTION =================
def detect_shape(cnt, mask_shape):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    if area < 1200 or peri == 0:
        return None, None

    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    corrected = np.zeros(mask_shape, dtype=np.uint8)

    if len(approx) == 3:
        cv2.polylines(corrected, [approx], True, 255, 4)
        return "Triangle", corrected

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        cv2.rectangle(corrected, (x, y), (x + w, y + h), 255, 4)
        return ("Square" if 0.9 < ar < 1.1 else "Rectangle"), corrected

    if area < 2000 and peri > 300:
        x1, y1 = cnt[0][0]
        x2, y2 = cnt[-1][0]
        cv2.line(corrected, (x1, y1), (x2, y2), 255, 4)
        return "Line", corrected

    circularity = 4 * math.pi * area / (peri * peri)

    if circularity > 0.82:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(corrected, (int(x), int(y)), int(r), 255, 4)
        return "Circle", corrected

    if 0.6 < circularity <= 0.82:
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(corrected, ellipse, 255, 4)
        return "Ellipse", corrected

    return None, None

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        x = int(hand.landmark[8].x * w)
        y = int(hand.landmark[8].y * h)

        if prev_x is None:
            prev_x, prev_y = x, y
        else:
            x = int(alpha * x + (1 - alpha) * prev_x)
            y = int(alpha * y + (1 - alpha) * prev_y)

            cv2.line(mask, (prev_x, prev_y), (x, y), 255, 4)
            cv2.line(canvas, (prev_x, prev_y), (x, y), (255,255,255), 4)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # ---------- SUBJECT PANEL ----------
    panel_y = frame.shape[0] - 35
    cv2.rectangle(output, (0, panel_y), (frame.shape[1], frame.shape[0]), (0,0,0), -1)

    cv2.putText(output,
                "1:Social  2:Science  3:Maths  4:General",
                (20, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(output,
                f"Subject: {current_subject}",
                (frame.shape[1]-240, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(output, f"Prediction: {prediction}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0,255,0), 3)

    cv2.imshow("Smart Air Drawing (Final)", output)

    key = cv2.waitKey(1) & 0xFF

    # ---------- SUBJECT SELECTION ----------
    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        current_subject = subjects[int(chr(key))]

    # ---------- PREDICT ----------
    elif key == ord('p'):
        kernel = np.ones((5,5), np.uint8)
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(clean, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            shape, corrected = detect_shape(cnt, mask.shape)

            if shape:
                prediction = shape
                canvas[:] = 0
                canvas[corrected > 0] = (255,255,255)

                img_name = map_shape_to_image(shape, current_subject)
                if img_name:
                    img = load_real_image(img_name)
                    if img is not None:
                        cv2.imshow("Real Image", cv2.resize(img, (300,300)))

    # ---------- CLEAR ----------
    elif key == ord('c'):
        canvas[:] = 0
        mask[:] = 0
        prediction = "None"

    # ---------- QUIT ----------
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()