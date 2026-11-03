
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🔬",
    layout="centered"
)

# Load CNN Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "skin_cancer_model.keras",
        compile=False
    )

model = load_model()

# Website Heading
st.title("🔬 Skin Cancer Classification")
st.subheader("Benign vs Malignant Detection")

st.write(
    "Upload a skin image to get a prediction "
    "from our CNN model."
)

st.divider()

# Image Upload
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    # Display Image
    st.image(
        image,
        caption="Uploaded Skin Image",
        use_container_width=True
    )

    # Predict Button
    if st.button("Predict"):

        with st.spinner("Analyzing Image..."):

            # Preprocessing
            image = image.resize((224, 224))

            image_array = np.array(image)
            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array, axis=0
            )

            # Model Prediction
            prediction = model.predict(
                image_array,
                verbose=0
            )

            probability = float(prediction[0][0])

            # Display Result
            st.subheader("Prediction Result")

            if probability > 0.5:

                st.error("Model Prediction: Malignant")

                st.write(
                    f"Model score: {probability * 100:.2f}%"
                )

            else:

                st.success("Model Prediction: Benign")

                st.write(
                    f"Model score: {(1-probability) * 100:.2f}%"
                )

st.divider()

# Disclaimer
st.warning(
    "Disclaimer: This is an educational AI project. "
    "It cannot diagnose skin cancer or replace a dermatologist. "
    "Please consult a qualified medical professional."
)

st.caption("Developed using CNN and Streamlit")
