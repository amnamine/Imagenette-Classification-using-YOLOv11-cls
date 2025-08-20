import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

from ultralytics import YOLO

# Load YOLOv11 classification model
model = YOLO("imagenette.pt")

class YOLOClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg="#2C2C2C")

        self.image_path = None
        self.img_label = None
        self.displayed_image = None

        # Title
        title = tk.Label(root, text="YOLOv11 Image Classifier", font=("Arial", 18, "bold"), fg="white", bg="#2C2C2C")
        title.pack(pady=15)

        # Buttons
        btn_frame = tk.Frame(root, bg="#2C2C2C")
        btn_frame.pack(pady=10)

        load_btn = tk.Button(btn_frame, text="Load Image", width=15, command=self.load_image, bg="#4CAF50", fg="white")
        load_btn.grid(row=0, column=0, padx=10)

        predict_btn = tk.Button(btn_frame, text="Predict", width=15, command=self.predict_image, bg="#2196F3", fg="white")
        predict_btn.grid(row=0, column=1, padx=10)

        reset_btn = tk.Button(btn_frame, text="Reset", width=15, command=self.reset, bg="#f44336", fg="white")
        reset_btn.grid(row=0, column=2, padx=10)

        # Image display
        self.img_panel = tk.Label(root, bg="#2C2C2C")
        self.img_panel.pack(pady=20)

        # Prediction result
        self.result_label = tk.Label(root, text="", font=("Arial", 14), fg="yellow", bg="#2C2C2C")
        self.result_label.pack(pady=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if self.image_path:
            pil_img = Image.open(self.image_path)
            pil_img.thumbnail((400, 400))
            self.displayed_image = ImageTk.PhotoImage(pil_img)
            self.img_panel.config(image=self.displayed_image)
            self.result_label.config(text="")  # Clear previous prediction

    def predict_image(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        results = model(self.image_path)  # Run YOLO classification
        probs = results[0].probs  # Probabilities
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)

        class_name = model.names[class_id]

        self.result_label.config(text=f"Prediction: {class_name} ({confidence:.2f})")

    def reset(self):
        self.image_path = None
        self.img_panel.config(image="")
        self.result_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOClassifierApp(root)
    root.mainloop()
