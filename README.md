# Tomato Disease Classification 🍅

A deep-learning image classification application that predicts the class of a tomato leaf image. The project includes the training notebook, image dataset, exported TensorFlow model, and a Streamlit web application.

<p align="center">
  <a href="https://tomato-diseaseclassification.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?style=for-the-badge&logo=streamlit" alt="Open the live Streamlit app">
  </a>
</p>

## Live application

**Try the classifier:** https://tomato-diseaseclassification.streamlit.app/

Upload a clear tomato leaf image in JPG, JPEG, or PNG format and select **Classify image**. The application displays the predicted class, the model's output score, and the top three predictions.

## Supported classes

The application predicts one of the following nine classes:

1. Tomato — Bacterial Spot
2. Tomato — Early Blight
3. Tomato — Late Blight
4. Tomato — Leaf Mold
5. Tomato — Septoria Leaf Spot
6. Tomato — Spider Mites
7. Tomato — Yellow Leaf Curl Virus
8. Tomato — Mosaic Virus
9. Tomato — Healthy

## Application features

- Upload tomato leaf images in JPG, JPEG, or PNG format.
- Convert images to RGB and resize them to the dimensions expected by the model.
- Run inference with the exported TensorFlow SavedModel.
- Display the predicted class and the three highest-scoring classes.
- Cache the loaded model with Streamlit to avoid reloading it on every interaction.
- Inspect model input details, raw scores, and input-sensitivity probes in the diagnostics section.

## Model and inference

The application uses a trained TensorFlow/Keras image-classification model. The training and experimentation notebook is available in [Tomato_Disease_Prediction.ipynb](Tomato_Disease_Prediction.ipynb).

The deployed application prefers the TensorFlow SavedModel at `model/1/`. The `tomato_model.keras` archive is retained as a fallback.

### Input preprocessing

The current SavedModel expects a batch of RGB images with an input shape of `(None, 256, 256, 3)`.

The inference pipeline is:

1. Open the uploaded image.
2. Convert it to RGB.
3. Resize it to 256 × 256 pixels.
4. Convert the pixel array to `float32`.
5. Pass pixel values in the **0–255** range to the model.

**Important:** The trained model contains a `Rescaling(1./255)` layer internally. Therefore, the application must not divide the uploaded image by 255 before inference. Doing so would normalize the image twice and can produce incorrect predictions.

The displayed model score is the model's output score, not a calibrated measure of the probability that a diagnosis is correct.

## Repository structure

```text
Tomato_Disease_Classification/
├── Tomato_Dataset/                 # Tomato leaf images grouped by class
├── Tomato_Disease_Prediction.ipynb # Training and experimentation notebook
├── model/
│   └── 1/
│       ├── saved_model.pb          # TensorFlow SavedModel graph
│       ├── fingerprint.pb
│       └── variables/              # Saved model weights
├── tomato_model.keras              # Keras model fallback
├── tomato_app.py                   # Streamlit application (deployment entry point)
├── main.py                         # Earlier application script
├── requirements.txt                # Python dependencies
└── README.md
```

Keep the complete `model/1/` directory, including its `variables/` files, in the repository. The SavedModel cannot be loaded correctly if its checkpoint files are missing.

## Technology stack

- **Python 3.11**
- **TensorFlow / TensorFlow CPU 2.13.1**
- **Keras**
- **Streamlit 1.35.0**
- **NumPy 1.24.3**
- **Pillow 10.4.0**
- **GitHub** for source control
- **Streamlit Community Cloud** for deployment

The exact dependency versions are listed in [requirements.txt](requirements.txt).

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/CodeWithRjUdit/Tomato_Disease_Classification.git
cd Tomato_Disease_Classification
```

### 2. Create and activate a virtual environment

Use Python 3.11.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Streamlit app

Run the deployment entry point:

```bash
streamlit run tomato_app.py
```

Streamlit will print a local URL in the terminal, usually http://localhost:8501.

## Deploy on Streamlit Community Cloud

The live application is deployed from this GitHub repository.

To create or reproduce the deployment:

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Connect your GitHub account.
3. Create an app from `CodeWithRjUdit/Tomato_Disease_Classification`.
4. Select the `main` branch.
5. Set **Main file path** to `tomato_app.py`.
6. In Advanced settings, select **Python 3.11**.
7. Deploy the application.

Use `tomato_app.py` as the entry point. The older `main.py` script is not the deployment entry point for this tomato-only application.

## Limitations and responsible use

- This project is an experimental image-classification demonstration, not a validated agricultural diagnostic system.
- Predictions can be affected by image quality, lighting, backgrounds, camera conditions, and differences between training images and real-world images.
- A high model score does not guarantee that a prediction is correct; the scores are not calibrated diagnostic probabilities.
- Confirm suspected plant diseases with a qualified agricultural expert before making treatment or crop-management decisions.

## License and reuse

No license is currently declared in this repository. Contact the repository owner before reusing, modifying, or redistributing the code, model, or dataset.

---

**Live demo:** https://tomato-diseaseclassification.streamlit.app/  
**Source code:** https://github.com/CodeWithRjUdit/Tomato_Disease_Classification
