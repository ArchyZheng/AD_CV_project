import torch
from dataset import get_train_or_val_dataloader
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
import wandb
import torchmetrics
from utlis import organize_output
from model import baseResnet
from dataset_fortensor import fashionDataset
from torch.utils.data import DataLoader


def train(model, dataloader, optimiser, criterion):
    model.train()
    epoch_loss = 0  # loss
    epoch_precision = 0  # metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for x, y in dataloader:
        optimiser.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat.softmax(dim=1), y.softmax(dim=1))
        y_hat_transform = organize_output(y_hat=y_hat, k=6)
        precision = torchmetrics.functional.precision(preds=y_hat_transform, target=y,
                                                      task="multilabel", num_labels=26)
        epoch_precision += precision
        epoch_loss += loss.item()

        loss.backward()
        optimiser.step()

    return epoch_loss / len(dataloader), epoch_precision / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0  # loss
    epoch_precision = 0  # metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat.softmax(dim=1), y.softmax(dim=1))
            y_hat_transform = organize_output(y_hat=y_hat, k=6)
            precision = torchmetrics.functional.precision(preds=y_hat_transform, target=y,
                                                          task="multilabel", num_labels=26)
            epoch_precision += precision
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader), epoch_precision / len(dataloader)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init()

    # batch_size = 16  # TODO: changed by wandb
    batch_size = wandb.config.batch_size  # TODO: changed by wandb
    # train_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/train.txt',
    #                                                label_list_dir='split/train_attr.txt', shuffle=True,
    #                                                batch_size=batch_size)
    #
    # val_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/val.txt',
    #                                              label_list_dir='split/val_attr.txt', shuffle=True,
    #                                              batch_size=batch_size)
    train_dataset = fashionDataset(feature_dir='train_tensor.pt', label_dir='train_label.pt')
    val_dataset = fashionDataset(feature_dir='val_tensor.pt', label_dir='val_label.pt')
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=1,
                                  batch_size=wandb.config.batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, num_workers=1, batch_size=wandb.config.batch_size)

    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = baseResnet()
    model = model.to(device)
    epochs = wandb.config.epochs  # TODO: changed by wandb
    # epochs = 50  # TODO: changed by wandb
    # TODO: optimizer dict, figure out all of the parameter which occur in optimizer
    optimizer = SGD(params=model.parameters(), lr=wandb.config.lr, momentum=0.9, weight_decay=wandb.config.wd)
    # optimizer = SGD(params=model.parameters(), lr=0.05, momentum=0.9, weight_decay=0)

    # TODO accuracy is metrics
    # criterion = nn.MultiLabelMarginLoss().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(epochs):
        train_loss, train_metric = train(model=model, criterion=criterion, optimiser=optimizer,
                                         dataloader=train_dataloader)
        val_loss, val_metric = evaluate(model=model, criterion=criterion,
                                        dataloader=val_dataloader)
        wandb.log(
            {"train_loss": train_loss, "train_metric": train_metric, "val_loss": val_loss, "val_metric": val_metric})
    wandb.finish()


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

