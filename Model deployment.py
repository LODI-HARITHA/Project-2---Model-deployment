import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Load the model
model = YOLO(r"C:/Users/DELL PC/Downloads/best (1).pt")  # Load the pretrained YOLOv8n model

# Function to display results

def display_results(results):
    for result in results:
        orig_img = result.orig_img  # Get the original PIL image

        # Convert PIL image to OpenCV format (BGR)
        img_cv = np.array(orig_img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
       
        # Draw bounding boxes and labels on the image
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = box.tolist()
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            
            # Get class label based on index in class_names list
            
            img_cv = cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box

        # Convert OpenCV image back to PIL format (RGB)
        img_pil = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        img_pil = Image.fromarray(img_pil)

        # Display the image
        st.image(img_pil, caption='Detected Image', use_column_width=True)


        # Get and display the detected object count
        object_count = len(result.boxes.data)
        st.write("Detected_Count:", object_count)





# Streamlit app
st.title("Count of steel rods")
upload = st.file_uploader(label="Upload Image:", type=['png', 'jpg', 'jpeg'])

if upload:
    img = Image.open(upload)
    results = model([img])  # Run inference on the uploaded image
    display_results(results)  # Display the results



