import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# ================= PIPE EDGE DETECTION ================= #

def detect_pipe_edges(frame):
    """
    Enhanced edge detection for rectangular / straight pipe structures
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast (important for industrial pipes)
    gray = cv2.equalizeHist(gray)

    # Reduce noise but keep edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Canny Edge Detection
    edges = cv2.Canny(
        blurred,
        threshold1=60,
        threshold2=180,
        L2gradient=True
    )

    # Morphological closing to strengthen pipe edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Detect straight lines (rectangular pipe emphasis)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=80,
        maxLineGap=10
    )

    output = frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return edges, output


# ================= IMAGE PROCESSING ================= #

def process_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Error loading image")
        return

    edges, highlighted = detect_pipe_edges(image)

    cv2.imshow("Original Image", image)
    cv2.imshow("Pipe Edges (Canny)", edges)
    cv2.imshow("Pipe Highlighted (Rectangular Detection)", highlighted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================= VIDEO / CAMERA PROCESSING ================= #

def process_video(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error opening video/camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges, highlighted = detect_pipe_edges(frame)

        cv2.imshow("Pipe Edges (Canny)", edges)
        cv2.imshow("Pipe Highlighted (Rectangular Detection)", highlighted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ================= FILE DIALOG ================= #

def upload_image():
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if path:
        process_image(path)


def upload_video():
    path = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if path:
        process_video(path)


def live_camera():
    process_video(0)


# ================= MAIN GUI ================= #

def main_window():
    root = tk.Tk()
    root.title("Pipe Edge Detection System")
    root.geometry("350x250")

    label = tk.Label(
        root,
        text="Select Input Source",
        font=("Arial", 16)
    )
    label.pack(pady=20)

    btn_image = tk.Button(
        root,
        text="Upload Image",
        width=20,
        height=2,
        command=upload_image
    )
    btn_image.pack(pady=5)

    btn_video = tk.Button(
        root,
        text="Upload Video",
        width=20,
        height=2,
        command=upload_video
    )
    btn_video.pack(pady=5)

    btn_camera = tk.Button(
        root,
        text="Live Camera Feed",
        width=20,
        height=2,
        command=live_camera
    )
    btn_camera.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main_window()
