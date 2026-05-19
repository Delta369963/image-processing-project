from src.datasets.dataloader import create_dataloaders

train_loader, val_loader = create_dataloaders()

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

for images, labels in train_loader:

    print("Image batch shape:", images.shape)

    print("Labels:", labels)

    break