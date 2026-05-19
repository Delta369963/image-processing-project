import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from src.configs.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    CHECKPOINTS_DIR
)

from src.datasets.dataloader import (
    create_dataloaders
)

from src.models.classification.resnet import (
    build_resnet18
)

# =========================================================
# TENSORBOARD WRITER
# =========================================================

writer = SummaryWriter(
    log_dir="logs/training"
)

# =========================================================
# CHECKPOINT DIRECTORY
# =========================================================

CHECKPOINTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# DATALOADERS
# =========================================================

train_loader, val_loader = create_dataloaders()

# =========================================================
# MODEL
# =========================================================

model = build_resnet18()

model = model.to(DEVICE)

# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# =========================================================
# LR SCHEDULER
# =========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

# =========================================================
# EARLY STOPPING
# =========================================================

patience = 5

epochs_without_improvement = 0

best_val_accuracy = 0.0

# =========================================================
# TRAINING LOOP
# =========================================================

for epoch in range(NUM_EPOCHS):

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()

    running_loss = 0.0

    train_correct = 0
    train_total = 0

    progress_bar = tqdm(train_loader)

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

        train_accuracy = (
            100 * train_correct / train_total
        )

        progress_bar.set_description(
            f"Epoch {epoch+1}/{NUM_EPOCHS}"
        )

        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=train_accuracy
        )

    epoch_loss = (
        running_loss / len(train_loader)
    )

    epoch_train_accuracy = (
        100 * train_correct / train_total
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    epoch_val_accuracy = (
        100 * val_correct / val_total
    )

    # =====================================================
    # SCHEDULER STEP
    # =====================================================

    scheduler.step(epoch_val_accuracy)

    # =====================================================
    # PRINT METRICS
    # =====================================================

    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {epoch_loss:.4f} "
        f"Train Acc: {epoch_train_accuracy:.2f}% "
        f"Val Acc: {epoch_val_accuracy:.2f}%"
    )

    # =====================================================
    # TENSORBOARD LOGGING
    # =====================================================

    writer.add_scalar(
        "Loss/Train",
        epoch_loss,
        epoch
    )

    writer.add_scalar(
        "Accuracy/Train",
        epoch_train_accuracy,
        epoch
    )

    writer.add_scalar(
        "Accuracy/Validation",
        epoch_val_accuracy,
        epoch
    )

    current_lr = optimizer.param_groups[0]["lr"]

    writer.add_scalar(
        "Learning_Rate",
        current_lr,
        epoch
    )

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if epoch_val_accuracy > best_val_accuracy:

        best_val_accuracy = epoch_val_accuracy

        epochs_without_improvement = 0

        checkpoint_path = (
            CHECKPOINTS_DIR /
            "best_resnet18.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"Best validation model saved: "
            f"{checkpoint_path}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement for "
            f"{epochs_without_improvement} epoch(s)"
        )

        if epochs_without_improvement >= patience:

            print("\nEarly stopping triggered.")

            break

# =========================================================
# CLOSE WRITER
# =========================================================

writer.close()

print("\nTraining Complete.")