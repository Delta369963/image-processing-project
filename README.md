# Wall Crack Detection AI

A deep learning based computer vision system for detecting structural wall cracks using Convolutional Neural Networks and Grad-CAM explainability.

The project combines:

* CNN classification
* Explainable AI visualisation
* FastAPI backend
* React frontend
* Real-time image inference

The system analyses uploaded wall images and predicts whether the wall is:

* Cracked
* Non-cracked

It also generates Grad-CAM heatmaps showing the regions the model focused on while making predictions.

---

# Features

## Deep Learning Classification

* ResNet18 transfer learning
* Fine-tuned on wall crack datasets
* Real-time inference

## Explainable AI

* Grad-CAM heatmap generation
* Visual interpretation of CNN attention

## Backend

* FastAPI REST API
* Image upload endpoint
* Prediction API
* Heatmap serving

## Frontend

* React + Vite
* TailwindCSS modern UI
* Dynamic predictions
* Live image preview
* Interactive dashboard

## Training Pipeline

* Dataset preprocessing
* Augmentation support
* Validation monitoring
* Early stopping
* Checkpoint saving

---

# Tech Stack

## Machine Learning

* PyTorch
* Torchvision
* TIMM
* OpenCV
* Albumentations
* NumPy
* Pillow

## Backend

* FastAPI
* Uvicorn

## Frontend

* React
* Vite
* TailwindCSS
* Framer Motion
* Axios

---

# Project Structure

```text
image-processing-project/

├── deployment/
│   └── api/
│       └── app.py

├── frontend/

├── scripts/

├── src/
│   ├── augmentation/
│   ├── configs/
│   ├── datasets/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   └── visualization/

├── tests/

├── requirements.txt

└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Delta369963/image-processing-project.git

cd image-processing-project
```

---

# Backend Setup

## Create Virtual Environment

```bash
python3 -m venv venv
```

## Activate Environment

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Frontend Setup

```bash
cd frontend

npm install
```

---

# Running the Backend

From project root:

```bash
python3 -m uvicorn deployment.api.app:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Inside frontend folder:

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5174
```

---

# Model Training

```bash
python3 -m src.training.train
```

---

# Evaluation

```bash
python3 -m src.evaluation.evaluate
```

---

# Prediction

```bash
python3 -m src.inference.predict --image path_to_image.jpg
```

---

# Grad-CAM Visualisation

```bash
python3 -m src.visualization.gradcam
```

Generated heatmaps are stored in:

```text
outputs/heatmaps/
```

---

# API Endpoint

## POST `/predict`

Upload an image for prediction.

### Response Example

```json
{
  "prediction": "cracked",
  "confidence": 92.41,
  "cracked_probability": 92.41,
  "non_cracked_probability": 7.59,
  "heatmap_path": "outputs/heatmaps/gradcam_xxxxx.jpg"
}
```

---

# Current Capabilities

* Wall crack classification
* Real-time API inference
* Explainable AI heatmaps
* Modern frontend dashboard
* CNN transfer learning pipeline

---

# Future Improvements

* Crack segmentation using U-Net
* Mobile deployment
* ONNX/TensorRT optimisation
* Video inference
* Multi-class defect detection
* Cloud deployment
* Real-world industrial datasets

---

# Author

Nikhil Sharma

---

# License

This project is intended for educational and research purposes.
