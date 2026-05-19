import torch

from src.configs.config import (
    DEVICE,
    CHECKPOINTS_DIR
)

from src.datasets.dataloader import (
    create_dataloaders
)

from src.models.classification.resnet import (
    build_resnet18
)

from src.evaluation.metrics import (
    compute_metrics
)

# =========================================================
# LOAD DATALOADER
# =========================================================

train_loader, val_loader = create_dataloaders()

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
# EVALUATION
# =========================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

# =========================================================
# METRICS
# =========================================================

results = compute_metrics(
    all_labels,
    all_predictions
)

# =========================================================
# DISPLAY
# =========================================================

print("\nEvaluation Results\n")

for key, value in results.items():

    print(f"{key}:\n{value}\n")