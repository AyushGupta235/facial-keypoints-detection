import torch
from data import FacialKeyPointDataset
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')
from efficientnet_pytorch import EfficientNet
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_rmse,
    get_submission
)

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device, epoch):
    losses = []
    tqdm_loader = tqdm(loader)
    num_examples = 0
    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        data = data.to(device=device, dtype = torch.float32)
        targets = targets.to(device=device, dtype = torch.float32)

        # forward pass
        scores = model(data)
        scores[targets==-1] = -1
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores[targets != -1])
        losses.append(loss.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss average on epoch {epoch}: {(sum(losses)/num_examples)**0.5}")

def main():
    train_ds = FacialKeyPointDataset(
        csv_file="data/train_15.csv",
        transform=config.train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    val_ds = FacialKeyPointDataset(
        csv_file="data/val_15.csv",
        transform=config.val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    test_ds = FacialKeyPointDataset(
        csv_file="data/test.csv",
        transform=config.val_transforms,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    loss_fn = nn.MSELoss(reduction="sum")
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(1280, 30)
    model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    model_4 = EfficientNet.from_pretrained("efficientnet-b0")
    model_4._fc = nn.Linear(1280, 30)
    model_15 = EfficientNet.from_pretrained("efficientnet-b0")
    model_15._fc = nn.Linear(1280, 30)
    model_4.to(config.DEVICE)
    model_15.to(config.DEVICE)

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)
        load_checkpoint(torch.load("b0_4.pth.tar"), model_4, optimizer, config.LEARNING_RATE)
        load_checkpoint(torch.load("b0_15.pth.tar"), model_15, optimizer, config.LEARNING_RATE)

    #get_submission(test_loader, test_ds, model_4=model_4, model_15=model_15)

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE, epoch)
        get_rmse(val_loader, model, loss_fn, config.DEVICE)

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

if __name__ == "__main__":
    main()
