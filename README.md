# Tomato Disease Classification

A Streamlit application that classifies tomato leaf images using a trained
TensorFlow/Keras model.

## Run locally

Use Python 3.11:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run tomato_app.py
```

## Deploy on Streamlit Community Cloud

1. Connect your GitHub account to Streamlit Community Cloud.
2. Create an app from this repository.
3. Select branch: `main`.
4. Set the app entry point to: `tomato_app.py`.
5. In Advanced settings, select Python 3.11.
6. Deploy.

The exported TensorFlow SavedModel under `model/1/` (including its
`variables/` checkpoint files) must remain in the repository. The
`tomato_model.keras` archive is retained as a fallback.

## Model input and preprocessing

The app reads the model's expected image dimensions from its input signature,
converts uploaded images to RGB, resizes them, and sends float32 pixel values
in the 0–255 range. The training notebook embeds a Keras Rescaling(1./255)
layer inside the model, so the app must not divide uploaded pixels by 255
before inference. Doing so would normalize the image twice.

## Important note

This is an experimental image-classification project. Model predictions may
be incorrect and are not a substitute for an expert agricultural assessment.
