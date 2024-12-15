import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import threading
import time

# --- Load the fine-tuned model ---
model = tf.keras.models.load_model('human_robot_classifier.h5')

# Image processing function to prepare the input for the model
def preprocess_image(frame):
    # Resize to match model input size (224, 224)
    img = cv2.resize(frame, (224, 224))
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize the image for MobileNetV2
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Preprocess for MobileNetV2
    return img

# Function to classify the image and return result
def classify_image(image_path):
    # Read the image from the file path
    img = cv2.imread(image_path)  # Load the image using OpenCV
    if img is None:
        return "Error: Unable to load image"
    
    img = preprocess_image(img)  # Preprocess the image
    predictions = model.predict(img)
    
    # Handle the case where model has two output classes
    predicted_class = 'Robot' if predictions[0][0] > 0.5 else 'Human'  # Adjust based on your model's output
    return predicted_class

# GUI for Image Upload Mode
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        result = classify_image(file_path)  # Pass the file path to classify_image
        result_label.config(text=f"Prediction: {result}", font=("Helvetica", 16))

# Real-Time Webcam Detection
def start_webcam():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Create a label to show webcam images on Tkinter interface
    label = tk.Label(root)
    label.pack(pady=10)

    def capture_frame():
        ret, frame = cap.read()
        if ret:
            # Process the frame to prepare for prediction
            img = preprocess_image(frame)  # Preprocess the frame
            
            # Classify the frame
            predictions = model.predict(img)
            
            # Handle predictions based on output shape
            predicted_class = 'Robot' if predictions[0][0] > 0.5 else 'Human'  # Adjust based on your model's output
            
            # Display the prediction on the frame
            cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to an image that Tkinter can display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

            # Update the label with the new frame
            label.config(image=img_tk)
            label.image = img_tk

            # Schedule the next frame update
            root.after(10, capture_frame)

    # Start capturing frames
    capture_frame()

    # Exit if the 'q' key is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_image(frame):
    """Capture the image and save it when the Space key is pressed."""
    # Save the captured image to disk
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"captured_image_{timestamp}.jpg"
    cv2.imwrite(file_name, frame)  # Save the captured frame as an image
    print(f"Captured and saved image: {file_name}")

# Create the main window
root = tk.Tk()
root.title("Human vs Robot Detection")
root.geometry("400x400")

# Title Label
title_label = tk.Label(root, text="Human or Robot Detection", font=("Helvetica", 18))
title_label.pack(pady=20)

# Buttons for Image and Webcam modes
image_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Helvetica", 14), width=20)
image_button.pack(pady=10)

webcam_button = tk.Button(root, text="Start Webcam", command=start_webcam, font=("Helvetica", 14), width=20)
webcam_button.pack(pady=10)

# Label to display the classification result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
