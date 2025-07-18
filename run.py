import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = 'mnist_cnn_model.h5'
IMAGE_SIZE = 28

# Load pretrained model
model = load_model(MODEL_PATH)

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28 pixels
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array

# Function to predict the digit and its confidence from the image
def predict_digit(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)  # Get the confidence (probability) of the prediction
    return predicted_digit, confidence, prediction

# Function to save the drawing as an image file
def save_image(image):
    # Save the image as a PNG file
    image_path = 'paint_output.png'
    image.save(image_path)
    return image_path

# Function to plot the confidence distribution
def plot_confidence(prediction):
    # Confidence for all digits
    confidence_scores = prediction[0]  # Assuming prediction is [batch_size, 10]
    digits = [str(i) for i in range(10)]

    plt.bar(digits, confidence_scores)
    plt.xlabel('Digit')
    plt.ylabel('Confidence')
    plt.title('Confidence Distribution')
    plt.show()

# Paint application class
class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Paint & Predict")
        self.root.geometry('500x500')

        # Create a Canvas to draw on
        self.canvas = tk.Canvas(self.root, width=300, height=300, bg='black')
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_painting)

        # Add buttons for actions
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_canvas)
        self.reset_button.pack(side=tk.LEFT, padx=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.save_button = tk.Button(self.root, text="Save", command=self.save_drawing)
        self.save_button.pack(side=tk.LEFT, padx=10)

        # Create a slider for brush thickness
        self.thickness_slider = tk.Scale(self.root, from_=1, to=20, orient="horizontal", label="Brush Thickness")
        self.thickness_slider.set(10)
        self.thickness_slider.pack(pady=10)

        # Label to display prediction and confidence
        self.prediction_label = tk.Label(self.root, text="Prediction: None", font=("Helvetica", 12))
        self.prediction_label.pack(pady=10)

        self.confidence_label = tk.Label(self.root, text="Confidence: None", font=("Helvetica", 12))
        self.confidence_label.pack(pady=10)

        # Progress bar to show confidence
        self.progress_bar = ttk.Progressbar(self.root, length=300, orient="horizontal", mode="determinate")
        self.progress_bar.pack(pady=10)

        self.prev_x = None
        self.prev_y = None
        self.drawing_image = Image.new('RGB', (300, 300), color='black')
        self.draw = ImageDraw.Draw(self.drawing_image)

    def paint(self, event):
        # Get the brush size from the slider
        brush_size = self.thickness_slider.get()
        if self.prev_x and self.prev_y:
            self.canvas.create_line(self.prev_x, self.prev_y, event.x, event.y,
                                    width=brush_size, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.prev_x, self.prev_y, event.x, event.y], fill='white', width=brush_size)
        self.prev_x = event.x
        self.prev_y = event.y

    def stop_painting(self, event):
        self.prev_x = None
        self.prev_y = None

    def reset_canvas(self):
        self.canvas.delete("all")
        self.drawing_image = Image.new('RGB', (300, 300), color='black')
        self.draw = ImageDraw.Draw(self.drawing_image)
        # Reset labels and progress bar
        self.prediction_label.config(text="Prediction: None")
        self.confidence_label.config(text="Confidence: None")
        self.progress_bar['value'] = 0

    def save_drawing(self):
        file_path = save_image(self.drawing_image)
        messagebox.showinfo("Saved", f"Image saved as {file_path}")

    def predict(self):
        # Save the image and run prediction
        image_path = save_image(self.drawing_image)
        image = Image.open(image_path)
        predicted_digit, confidence, prediction = predict_digit(image)

        # Update labels with prediction and confidence
        self.prediction_label.config(text=f"Prediction: {predicted_digit}")
        self.confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")
        self.progress_bar['value'] = confidence * 100  # Update progress bar with confidence

        # Show a message box with the prediction and confidence
        messagebox.showinfo("Prediction", f"Predicted digit: {predicted_digit}\nConfidence: {confidence * 100:.2f}%")

        # Plot the confidence distribution
        plot_confidence(prediction)

        # Provide feedback based on confidence
        if confidence > 0.85:
            messagebox.showinfo("Feedback", "Great! The model is confident in your drawing!")
        elif confidence > 0.70:
            messagebox.showinfo("Feedback", "Good job! But you can try drawing a bit more clearly.")
        else:
            messagebox.showinfo("Feedback", "Confidence is low. Try drawing a clearer digit.")

if __name__ == "__main__":
    # Initialize the root Tkinter window
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
