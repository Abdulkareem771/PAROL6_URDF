import cv2
import numpy as np
import csv
from tkinter import Tk, filedialog, Button, Label

# ---------------------------
# CONFIGURATION
# ---------------------------
CHESSBOARD_SIZE = (11, 9)   # inner corners (cols, rows)
CSV_OUTPUT = "chessboard_corners.csv"

# ---------------------------
# GLOBALS
# ---------------------------
img = None
corners = None
window_name = "Chessboard Corners"

# ---------------------------
# EXPORT TO CSV
# ---------------------------
def export_to_csv(corners):
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corner_index", "x_pixel", "y_pixel"])
        for i, c in enumerate(corners):
            x, y = c.ravel()
            writer.writerow([i, int(x), int(y)])
    print(f"[OK] CSV saved: {CSV_OUTPUT}")

# ---------------------------
# MOUSE CALLBACK
# ---------------------------
def mouse_callback(event, x, y, flags, param):
    global img, corners

    if event == cv2.EVENT_LBUTTONDOWN and corners is not None:
        d = np.linalg.norm(corners.reshape(-1, 2) - np.array([x, y]), axis=1)
        idx = np.argmin(d)
        cx, cy = corners[idx].ravel().astype(int)

        vis = img.copy()
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"Corner {idx}: ({cx}, {cy})",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        cv2.imshow(window_name, vis)

# ---------------------------
# LOAD IMAGE
# ---------------------------
def load_image():
    global img, corners

    path = filedialog.askopenfilename(
        title="Select Chessboard Image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
    )

    if not path:
        return

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret:
        print("[ERROR] Chessboard not detected")
        cv2.imshow(window_name, img)
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)

    export_to_csv(corners)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, img)

# ---------------------------
# TKINTER GUI
# ---------------------------
root = Tk()
root.title("Chessboard Corner Detector")
root.geometry("300x120")

Label(root, text="Chessboard Corner Detector", font=("Arial", 12)).pack(pady=10)

Button(root, text="Browse Image", width=20, command=load_image).pack()

# ---------------------------
# MAIN LOOP (CRITICAL FIX)
# ---------------------------
while True:
    root.update_idletasks()
    root.update()

    if cv2.waitKey(1) & 0xFF == 27:  # ESC closes OpenCV window
        break

cv2.destroyAllWindows()
root.destroy()
