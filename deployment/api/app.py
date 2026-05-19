from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os
import uuid

from src.inference.predict import (
    predict_image
)

from src.visualization.gradcam import (
    generate_gradcam
)

app = FastAPI()

app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)

# =========================================================
# ROOT
# =========================================================

@app.get("/")

def home():

    return {
        "message":
        "Wall Crack Detection API Running"
    }

# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")

async def predict(
    file: UploadFile = File(...)
):

    # =====================================================
    # SAVE TEMP IMAGE
    # =====================================================

    temp_filename = (
        f"temp_{uuid.uuid4()}.jpg"
    )

    with open(
        temp_filename,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    results = predict_image(
        temp_filename
    )

    # =====================================================
    # GENERATE GRADCAM
    # =====================================================

    heatmap_path = generate_gradcam(
        temp_filename
    )

    if os.path.exists(temp_filename):

        os.remove(temp_filename)

    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "prediction":
            results["prediction"],

        "confidence":
            round(
                results["confidence"],
                2
            ),

        "cracked_probability":
            round(
                results[
                    "cracked_probability"
                ],
                2
            ),

        "non_cracked_probability":
            round(
                results[
                    "non_cracked_probability"
                ],
                2
            ),

        "heatmap_path":
            heatmap_path
    }