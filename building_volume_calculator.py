import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import tempfile

# Create a Streamlit app
st.title("Building Component Extractor")

# Add a file uploader
uploaded_file = st.file_uploader("Upload an image or PDF file", type=["jpg", "png", "pdf"])

# Add a button to extract the building components
if st.button("Extract Building Components"):
    # Load the uploaded file
    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        if uploaded_file.type == "application/pdf":
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = f"{tmpdir}/uploaded_file.pdf"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                images = convert_from_path(file_path)
                img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imdecode(uploaded_file.read(), cv2.IMREAD_COLOR)

        # Preprocess the image
        img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Find contours of the building components
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract dimensions and calculate volumes
        dimensions = []
        volumes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            dimensions.append((w, h, x, y, area, perimeter))
            volume = w * h * x  # Simple rectangular prism volume calculation
            volumes.append(volume)

        # Organize the data into a Pandas DataFrame
        data = pd.DataFrame({'Dimensions': dimensions, 'Volumes': volumes})

        # Display the output
        st.write("Dimensions and Volumes:")
        st.write(data)