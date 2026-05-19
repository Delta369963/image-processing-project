import cv2
from pathlib import Path

from src.configs.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    IMAGE_SIZE
)

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# IMAGE PREPROCESSING FUNCTION
# =========================================================

def preprocess_image(image_path, output_path):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Failed to load: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(f"Processed: {output_path}")

# =========================================================
# PROCESS ALL IMAGES
# =========================================================

def process_dataset():

    image_extensions = [".jpg", ".jpeg", ".png"]

    for image_path in RAW_DATA_DIR.rglob("*"):

        if image_path.suffix.lower() not in image_extensions:
            continue

        relative_path = image_path.relative_to(RAW_DATA_DIR)

        output_path = PROCESSED_DATA_DIR / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        preprocess_image(image_path, output_path)

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    process_dataset()