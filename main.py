import torch
from dataset import get_train_or_val_dataloader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
import torchvision.transforms as T
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from model import baseResnet
import wandb
import yaml


def main():
    wandb.init()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = wandb.config.batch_size  # TODO: changed by wandb
    train_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/train.txt',
                                                   label_list_dir='split/train_attr.txt', shuffle=True,
                                                   batch_size=batch_size)

    val_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/val.txt',
                                                 label_list_dir='split/val_attr.txt', shuffle=True,
                                                 batch_size=batch_size)
    transform = nn.Sequential(
        T.Resize((224, 224))
    ).to(device)
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = baseResnet()
    model.to(device)
    epochs = wandb.config.epochs  # TODO: changed by wandb
    # TODO: optimizer dict
    optimizer = SGD(params=model.parameters(), lr=wandb.config.lr, momentum=0.9, weight_decay=wandb.config.wd)
    # TODO: scheduler dict
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    criterion = nn.MultiLabelMarginLoss().to(device)
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        val_loss = 0
        val_acc = 0
        for X, y in train_dataloader:
            model.train()
            optimizer.zero_grad()
            X.to(device)
            y.to(device)
            y_hat = model(input)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            train_loss += batch_size * loss.item()
            # the loss function will automatically use mean operation after calculate.
        wandb.log({'train_loss': train_loss})


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--cfg', type=str, default='config.yaml')
    parser.add_argument('--prj_name', type=str)
    args = parser.parse_args()

    with open(args.cfg, 'r') as stream:
        sweep_configuration = yaml.safe_load(stream)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.prj_name)
    wandb.agent(sweep_id=sweep_id, function=main)
