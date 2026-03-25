import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk

class YoloGuiTester:
    def __init__(self, window, model_path):
        self.window = window
        self.window.title("Welding Seam Segmentation Tester")
        
        # Load your trained model
        self.model = YOLO(model_path)
        
        # GUI Elements
        self.btn_load = tk.Button(window, text="Select Image", command=self.load_and_predict)
        self.btn_load.pack(pady=10)
        
        self.canvas = tk.Label(window)
        self.canvas.pack(padx=10, pady=10)

    def load_and_predict(self):
        # 1. Open File Dialog
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # 2. Run Inference
        # We use a low confidence threshold initially to see what the model captures
        results = self.model.predict(source=file_path, conf=0.25, save=False)

        # 3. Process Result
        for r in results:
            # .plot() draws the masks and boxes onto the image
            im_bgr = r.plot() 
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            
            # 4. Resize for GUI display if the image is too large
            img = Image.fromarray(im_rgb)
            img.thumbnail((800, 600)) 
            
            # 5. Update GUI Label
            tk_img = ImageTk.PhotoImage(img)
            self.canvas.config(image=tk_img)
            self.canvas.image = tk_img

# --- Start the App ---
if __name__ == "__main__":
    root = tk.Tk()
    # Make sure 'best.pt' is in the same folder as this script
    app = YoloGuiTester(root, "best.pt")
    root.mainloop()