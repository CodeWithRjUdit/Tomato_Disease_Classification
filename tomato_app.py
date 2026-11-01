from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="🍅",
    layout="centered",
)

MODEL_PATH = Path(__file__).resolve().parent / "tomato_model.keras"

TOMATO_CLASSES = [
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy",
]


@st.cache_resource(show_spinner="Loading the tomato disease model...")
def load_model():
    """Load a complete inference model and verify its class-output dimension."""
    saved_model_dir = Path(__file__).resolve().parent / "model" / "1"
    keras_model_path = Path(__file__).resolve().parent / "tomato_model.keras"

    errors = []

    # The repository also contains a TensorFlow SavedModel export. Prefer it
    # because the .keras archive has been failing to restore Conv2D weights.
    if (saved_model_dir / "saved_model.pb").is_file():
        try:
            model = tf.keras.models.load_model(saved_model_dir, compile=False)
            output_shape = model.output_shape

            if isinstance(output_shape, list):
                if len(output_shape) != 1:
                    raise ValueError(
                        "The SavedModel has multiple outputs; one output is expected."
                    )
                output_shape = output_shape[0]

            if output_shape[-1] != len(TOMATO_CLASSES):
                raise ValueError(
                    f"SavedModel outputs {output_shape[-1]} classes, but the app "
                    f"defines {len(TOMATO_CLASSES)} tomato classes."
                )

            return model
        except Exception as error:
            errors.append(f"SavedModel ({saved_model_dir}): {error}")

    # Keep the .keras archive as a fallback, but do not use it unless its
    # variables load successfully and its output dimension matches the labels.
    if keras_model_path.is_file():
        try:
            model = tf.keras.models.load_model(keras_model_path, compile=False)
            output_shape = model.output_shape

            if isinstance(output_shape, list):
                if len(output_shape) != 1:
                    raise ValueError(
                        "The .keras model has multiple outputs; one output is expected."
                    )
                output_shape = output_shape[0]

            if output_shape[-1] != len(TOMATO_CLASSES):
                raise ValueError(
                    f".keras model outputs {output_shape[-1]} classes, but the app "
                    f"defines {len(TOMATO_CLASSES)} tomato classes."
                )

            return model
        except Exception as error:
            errors.append(f"Keras archive ({keras_model_path.name}): {error}")

    detail = "\\n\\n".join(errors) if errors else (
        "Neither model/1/saved_model.pb nor tomato_model.keras was found."
    )
    raise RuntimeError(
        "No compatible tomato model could be loaded. "
        "The app will not make predictions using missing or uninitialized weights. "
        f"\\n\\nDetails:\\n{detail}"
    )

def get_model_image_size(model):
    """Read the model's expected image dimensions instead of hard-coding them."""
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError("This app expects a model with exactly one image input.")
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            f"Expected a 4D image input (batch, height, width, channels); "
            f"received {input_shape}."
        )

    _, height, width, channels = input_shape

    if channels not in (3, None):
        raise ValueError(
            f"The model expects {channels} channels. This app prepares RGB images."
        )

    # Keep a fallback for models with dynamic spatial dimensions.
    height = int(height) if height is not None else 256
    width = int(width) if width is not None else 256

    return (width, height)


def predict_disease(model, uploaded_image):
    """Preprocess an image and return its predicted tomato class and scores."""
    target_size = get_model_image_size(model)

    image = uploaded_image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    raw_scores = model.predict(image_batch, verbose=0)

    if isinstance(raw_scores, (list, tuple)):
        if len(raw_scores) != 1:
            raise ValueError("The model returned multiple outputs; one is expected.")
        raw_scores = raw_scores[0]

    scores = np.asarray(raw_scores, dtype=np.float64)

    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError(f"Unexpected model output shape: {scores.shape}.")

    scores = scores[0]

    if len(scores) != len(TOMATO_CLASSES):
        raise ValueError(
            f"The model returned {len(scores)} scores, but the app has "
            f"{len(TOMATO_CLASSES)} class labels. The label list must match "
            "the model's training class order."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction scores.")

    # Use the model's probabilities directly when they are already normalized.
    # Otherwise, interpret the returned values as logits.
    if (
        np.all(scores >= 0.0)
        and np.all(scores <= 1.0)
        and np.isclose(np.sum(scores), 1.0, atol=0.05)
    ):
        probabilities = scores
    else:
        probabilities = tf.nn.softmax(scores).numpy()

    predicted_index = int(np.argmax(probabilities))
    return predicted_index, probabilities


# ---------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------
st.title("🍅 Tomato Disease Classifier")
st.write(
    "Upload a tomato leaf image to get a prediction from the trained "
    "deep-learning model."
)

st.info(
    "For best results, upload a clear image in which the tomato leaf is "
    "visible and well lit."
)

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption="Uploaded tomato leaf",
            use_column_width=True,
        )

        if st.button("Classify image", type="primary", use_container_width=True):
            with st.spinner("Analysing the image..."):
                model = load_model()
                predicted_index, probabilities = predict_disease(model, image)

            predicted_class = TOMATO_CLASSES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100.0

            st.subheader("Prediction")
            st.success(predicted_class)
            st.metric("Model confidence", f"{confidence:.2f}%")

            st.subheader("Top predictions")
            top_indices = np.argsort(probabilities)[::-1][:3]

            for index in top_indices:
                label = TOMATO_CLASSES[int(index)]
                score = float(probabilities[index]) * 100.0
                st.write(f"**{label}** — {score:.2f}%")
                st.progress(min(max(float(probabilities[index]), 0.0), 1.0))

    except Exception as error:
        st.error("The image could not be processed.")
        st.caption(f"Technical details: {error}")

st.divider()
st.caption(
    "This is an experimental image-classification project. Predictions can be "
    "incorrect and should not be treated as a definitive plant-health diagnosis."
)
