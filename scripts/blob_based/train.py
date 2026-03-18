import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blob_based.dataset.config import *
from blob_based.dataset.bbox_dataset import BboxDataset
from blob_based.models.blob_cnn import BlobCNN


def main():
    print("\n=== INITIALIZING TRAINING ===\n")

    # 1️⃣ Load datasets (this is the slow NDVI + blob phase)
    train_ds = BboxDataset(f"{SPLIT_DIR}/train.txt")
    val_ds   = BboxDataset(f"{SPLIT_DIR}/val.txt")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train blobs: {len(train_ds)}")
    print(f"Val blobs:   {len(val_ds)}\n")

    # 2️⃣ Model, optimizer, loss
    model = BlobCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    # 3️⃣ Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{EPOCHS}]",
            leave=False
        )

        for imgs, labels in loop:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train loss = {avg_loss:.4f}")

    # 4️⃣ Save model
    torch.save(model.state_dict(), "blob_cnn.pth")
    print("\nModel saved as blob_cnn.pth\n")


if __name__ == "__main__":
    main()

	
