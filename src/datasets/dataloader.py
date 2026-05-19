from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.configs.config import (
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
    CLASS_NAMES
)

from src.utils.transforms import (
    train_transforms,
    val_transforms
)

# =========================================================
# CUSTOM DATASET
# =========================================================

class WallDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    image_paths = []
    labels = []

    for label_index, class_name in enumerate(CLASS_NAMES):

        class_dir = PROCESSED_DATA_DIR / class_name

        for image_path in class_dir.glob("*.jpg"):

            image_paths.append(image_path)

            labels.append(label_index)

    return image_paths, labels

# =========================================================
# CREATE DATALOADERS
# =========================================================

def create_dataloaders():

    image_paths, labels = load_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = WallDataset(
        X_train,
        y_train,
        transform=train_transforms
    )

    val_dataset = WallDataset(
        X_val,
        y_val,
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader