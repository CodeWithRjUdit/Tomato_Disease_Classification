import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load models
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

tomato_model = load_model("tomato_model.keras")
potato_model = load_model("potato.keras")

# Define prediction function
def predict(model, img, class_names, target_size):
    # Preprocess the image
    img = img.resize(target_size)  # Resize to model's input size
    img_array = np.array(img)  # Convert to array using np.array directly
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img_array)

    # Debug output: Check the shape and values of predictions
    st.write(f"Predictions: {predictions}")
    st.write(f"Prediction shape: {predictions.shape}")

    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

# Define class names
tomato_classes = [
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight", 
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites", 
    "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus", 
    "Tomato - Healthy"
]

potato_classes = [
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy"
]

# Streamlit App
st.title("Vegetable Disease Prediction")
st.write("Select a vegetable and upload an image to predict the disease.")

# Sidebar
st.sidebar.header("Select Vegetable")
vegetable = st.sidebar.selectbox("Choose a vegetable:", ["Tomato", "Potato"])

# Image upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load the image
    image = Image.open(uploaded_image)
    
    # Resize image for display
    display_image = image.copy()
    max_display_size = (400, 400)  # Limit display size to 400x400
    display_image.thumbnail(max_display_size)

    # Display resized image
    st.image(display_image, caption="Uploaded Image (Resized for Display)", use_column_width=False)
    
    # Predict disease
    if st.button("Predict"):
        target_size = (256, 256)  # Adjust to the model's expected input size (Change if required)
        if vegetable == "Tomato":
            predicted_class, confidence = predict(tomato_model, image, tomato_classes, target_size)
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence}%")
        elif vegetable == "Potato":
            predicted_class, confidence = predict(potato_model, image, potato_classes, target_size)
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence}%")
