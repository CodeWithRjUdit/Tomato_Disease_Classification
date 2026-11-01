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

The trained model file, `tomato_model.keras`, must remain in the repository root.

## Model input and preprocessing

The app reads the model's expected image dimensions from its input shape,
converts uploaded images to RGB, resizes them, and scales pixel values to
the range [0, 1], matching the preprocessing used in the original app.

## Important note

This is an experimental image-classification project. Model predictions may
be incorrect and are not a substitute for an expert agricultural assessment.
