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
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy",
]

@st.cache_resource(show_spinner="Loading the tomato disease model...")
def load_model():
    """Load the exported SavedModel signature, with a validated Keras fallback."""
    saved_model_dir = Path(__file__).resolve().parent / "model" / "1"
    keras_model_path = Path(__file__).resolve().parent / "tomato_model.keras"
    errors = []

    # Prefer the TensorFlow SavedModel export. tf.saved_model.load returns a
    # Trackable object, so inference must use its concrete serving signature.
    if (saved_model_dir / "saved_model.pb").is_file():
        try:
            loaded = tf.saved_model.load(str(saved_model_dir))
            signatures = loaded.signatures

            if not signatures:
                raise ValueError("SavedModel contains no callable signatures.")

            if "serving_default" in signatures:
                serving_fn = signatures["serving_default"]
            elif len(signatures) == 1:
                serving_fn = next(iter(signatures.values()))
            else:
                raise ValueError(
                    "SavedModel has no serving_default signature and exposes "
                    f"multiple signatures: {list(signatures.keys())}"
                )

            positional_specs, keyword_specs = serving_fn.structured_input_signature

            if keyword_specs and not positional_specs and len(keyword_specs) == 1:
                input_style = "keyword"
                input_name, input_spec = next(iter(keyword_specs.items()))
            elif len(positional_specs) == 1 and not keyword_specs:
                input_style = "positional"
                input_name = None
                input_spec = positional_specs[0]
            else:
                raise ValueError(
                    "Expected exactly one image input in the SavedModel signature. "
                    f"Received positional={positional_specs}, keyword={keyword_specs}."
                )

            output_specs = serving_fn.structured_outputs
            if isinstance(output_specs, dict):
                if len(output_specs) != 1:
                    raise ValueError(
                        "Expected exactly one prediction output. "
                        f"Received outputs: {list(output_specs.keys())}."
                    )
                output_name, output_spec = next(iter(output_specs.items()))
            else:
                output_name = None
                output_spec = output_specs

            input_shape = tuple(input_spec.shape)
            output_shape = tuple(output_spec.shape)

            if len(input_shape) != 4:
                raise ValueError(
                    f"Expected image input shape [batch, height, width, channels]; "
                    f"received {input_shape}."
                )
            if len(output_shape) != 2 or output_shape[-1] != len(TOMATO_CLASSES):
                raise ValueError(
                    f"SavedModel output shape is {output_shape}; expected "
                    f"[batch, {len(TOMATO_CLASSES)}] for the dataset's "
                    f"{len(TOMATO_CLASSES)} classes."
                )

            return {
                "kind": "saved_model",
                "signature": serving_fn,
                "input_style": input_style,
                "input_name": input_name,
                "input_shape": input_shape,
                "input_dtype": input_spec.dtype,
                "output_name": output_name,
            }
        except Exception as error:
            errors.append(f"SavedModel ({saved_model_dir}): {error}")

    # Fallback only if the .keras archive genuinely restores all its weights.
    if keras_model_path.is_file():
        try:
            keras_model = tf.keras.models.load_model(
                keras_model_path, compile=False
            )
            output_shape = keras_model.output_shape

            if isinstance(output_shape, list):
                if len(output_shape) != 1:
                    raise ValueError("Keras model has multiple outputs.")
                output_shape = output_shape[0]

            if output_shape[-1] != len(TOMATO_CLASSES):
                raise ValueError(
                    f"Keras model outputs {output_shape[-1]} classes, but "
                    f"{len(TOMATO_CLASSES)} dataset classes are configured."
                )

            return {
                "kind": "keras",
                "model": keras_model,
                "input_shape": tuple(keras_model.input_shape),
            }
        except Exception as error:
            errors.append(f"Keras archive ({keras_model_path.name}): {error}")

    details = "

".join(errors) if errors else (
        "Neither model/1/saved_model.pb nor tomato_model.keras was found."
    )
    raise RuntimeError(
        "No compatible tomato model could be loaded. "
        "The app will not make predictions using missing or uninitialized weights. "
        f"

Details:
{details}"
    )


def get_model_image_size(model_info):
    """Read the model's expected image dimensions."""
    input_shape = model_info["input_shape"]

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

    height = int(height) if height is not None else 256
    width = int(width) if width is not None else 256
    return (width, height)


def predict_disease(model_info, uploaded_image):
    """Preprocess an image and run either Keras or SavedModel inference."""
    target_size = get_model_image_size(model_info)

    image = uploaded_image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    if model_info["kind"] == "saved_model":
        input_tensor = tf.convert_to_tensor(
            image_batch, dtype=model_info["input_dtype"]
        )
        serving_fn = model_info["signature"]

        if model_info["input_style"] == "keyword":
            outputs = serving_fn(**{model_info["input_name"]: input_tensor})
        else:
            outputs = serving_fn(input_tensor)

        output_name = model_info["output_name"]
        raw_scores = outputs[output_name] if output_name is not None else outputs
    else:
        raw_scores = model_info["model"].predict(image_batch, verbose=0)

    if isinstance(raw_scores, (list, tuple)):
        if len(raw_scores) != 1:
            raise ValueError("The model returned multiple outputs; one is expected.")
        raw_scores = raw_scores[0]

    if hasattr(raw_scores, "numpy"):
        raw_scores = raw_scores.numpy()

    scores = np.asarray(raw_scores, dtype=np.float64)

    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError(f"Unexpected model output shape: {scores.shape}.")

    scores = scores[0]

    if len(scores) != len(TOMATO_CLASSES):
        raise ValueError(
            f"The model returned {len(scores)} scores, but the app has "
            f"{len(TOMATO_CLASSES)} dataset labels."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction scores.")

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
