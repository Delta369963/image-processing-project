import kagglehub
import shutil
import random

from pathlib import Path

# =========================================================
# PROJECT PATHS
# =========================================================

project_root = Path(
    __file__
).resolve().parent.parent

raw_dir = (
    project_root /
    "data/raw"
)

cracked_dir = (
    raw_dir /
    "cracked"
)

non_cracked_dir = (
    raw_dir /
    "non_cracked"
)

# =========================================================
# CLEAR OLD DATA
# =========================================================

for folder in [cracked_dir, non_cracked_dir]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in folder.glob("*"):

        file.unlink()

# =========================================================
# DATASET CONFIG
# =========================================================

datasets = [

    {
        "name":
        "arnavr10880/concrete-crack-images-for-classification",

        "positive":
        "Positive",

        "negative":
        "Negative"
    }

]

# =========================================================
# DOWNLOAD + MERGE
# =========================================================

crack_count = 0
non_crack_count = 0

for dataset in datasets:

    print(
        f"\nDownloading: {dataset['name']}"
    )

    dataset_path = kagglehub.dataset_download(
        dataset["name"]
    )

    source_path = Path(dataset_path)

    positive_dir = (
        source_path /
        dataset["positive"]
    )

    negative_dir = (
        source_path /
        dataset["negative"]
    )

    # =====================================================
    # CRACKED IMAGES
    # =====================================================

    positive_images = list(
        positive_dir.glob("*.jpg")
    )

    random.shuffle(
        positive_images
    )

    for image_path in positive_images[:1000]:

        destination = (
            cracked_dir /
            f"crack_{crack_count}.jpg"
        )

        shutil.copy(
            image_path,
            destination
        )

        crack_count += 1

    # =====================================================
    # NON-CRACKED IMAGES
    # =====================================================

    negative_images = list(
        negative_dir.glob("*.jpg")
    )

    random.shuffle(
        negative_images
    )

    for image_path in negative_images[:1000]:

        destination = (
            non_cracked_dir /
            f"wall_{non_crack_count}.jpg"
        )

        shutil.copy(
            image_path,
            destination
        )

        non_crack_count += 1

print("\nDataset expansion complete.")

print(
    f"Cracked images: {crack_count}"
)

print(
    f"Non-cracked images: {non_crack_count}"
)