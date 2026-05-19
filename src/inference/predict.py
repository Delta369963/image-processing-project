import argparse

import torch

from PIL import Image

from src.configs.config import (
    DEVICE,
    CLASS_NAMES,
    CHECKPOINTS_DIR
)

from src.models.classification.resnet import (
    build_resnet18
)

from src.utils.transforms import (
    val_transforms
)

# =========================================================
# LOAD MODEL
# =========================================================

model = build_resnet18()

checkpoint_path = (
    CHECKPOINTS_DIR /
    "best_resnet18.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

# =========================================================
# PREDICTION FUNCTION
# =========================================================

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = val_transforms(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        crack_probability = probabilities[
            0,
            0
        ].item()

        non_crack_probability = probabilities[
            0,
            1
        ].item()

        threshold = 0.40

        if crack_probability > threshold:

            prediction = "cracked"

            confidence_score = (
                crack_probability * 100
            )

        else:

            prediction = "non_cracked"

            confidence_score = (
                non_crack_probability * 100
            )

    return {

        "prediction":
            prediction,

        "confidence":
            confidence_score,

        "cracked_probability":
            crack_probability * 100,

        "non_cracked_probability":
            non_crack_probability * 100
    }

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    prediction, confidence = predict_image(
        args.image
    )

    print("\nPrediction Results\n")

    print(f"Predicted Class: {prediction}")

    print(f"Confidence: {confidence:.2f}%")