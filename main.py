import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
import wandb
import torchmetrics
from utlis import organize_output
from model import baseResnet
from dataset_fortensor import fashionDataset
from torch.utils.data import DataLoader
from utlis import formal_output
import numpy as np
from dataset_test import fashionDataset_1
from torchvision import transforms as T


def train(model, dataloader, optimiser, criterion, transforms, common):
    model.train()
    epoch_loss = 0  # loss
    epoch_precision = 0  # metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for x, y in dataloader:
        optimiser.zero_grad()
        x = x.to(device)
        color = T.RandomApply(T.ColorJitter(brightness=.5, hue=.3), p=0.5)
        x = color(x)
        x = common(x)
        x = transforms(x)
        y = y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        y_hat_transform = organize_output(y_hat=y_hat, k=6)
        precision = torchmetrics.functional.precision(preds=y_hat_transform, target=y,
                                                      task="multilabel", num_labels=26)
        epoch_precision += precision
        epoch_loss += loss.item()

        loss.backward()
        optimiser.step()

    return epoch_loss / len(dataloader), epoch_precision / len(dataloader)


def evaluate(model, dataloader, criterion, transforms):
    model.eval()
    epoch_loss = 0  # loss
    epoch_precision = 0  # metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x = transforms(x)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            y_hat_transform = organize_output(y_hat=y_hat, k=6)
            precision = torchmetrics.functional.precision(preds=y_hat_transform, target=y,
                                                          task="multilabel", num_labels=26)
            epoch_precision += precision
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader), epoch_precision / len(dataloader)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init()

    batch_size = wandb.config.batch_size  # TODO: changed by wandb
    train_dataset = fashionDataset(feature_dir='train_tensor.pt', label_dir='train_label.pt')
    val_dataset = fashionDataset(feature_dir='val_tensor.pt', label_dir='val_label.pt')
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=1,
                                  batch_size=wandb.config.batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, num_workers=1, batch_size=wandb.config.batch_size)

    model = baseResnet(baseModel=wandb.config.baseModel)
    model = model.to(device)
    epochs = wandb.config.epochs  # TODO: changed by wandb
    # TODO: optimizer dict, figure out all of the parameter which occur in optimizer
    if wandb.config.optimizer == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.wd)
    else:
        optimizer = SGD(params=model.parameters(), lr=wandb.config.lr, momentum=wandb.config.momentum,
                        weight_decay=wandb.config.wd)
    if wandb.config.scheduler == "Cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, wandb.config.epochs)
    else:
        scheduler = lr_scheduler.ConstantLR(optimizer=optimizer, factor=1.0, total_iters=wandb.config.epochs)

    # TODO accuracy is metrics
    # criterion = nn.MultiLabelMarginLoss().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    # TODO: add weight for each class
    weight = torch.ones(size=(1, 26))
    if wandb.config.weight:
        weight[0, [0 + 4, 7 + 1, 10 + 0, 13 + 2, 17 + 4, 23 + 1]] = wandb.config.weight_value
    criterion = nn.BCEWithLogitsLoss(weight=weight).to(device)

    transforms_train = T.RandomApply(nn.Sequential(
        T.Pad(padding=2),
        T.RandomHorizontalFlip(),
        # T.ColorJitter(brightness=.5, hue=.3),
        T.RandomCrop(size=(224, 224)),
        T.Resize(size=(wandb.config.resolutionrate, wandb.config.resolutionrate)),
        T.Resize(size=(224, 224))
    ), p=wandb.config.trans_probability).to(device)
    transforms_test = nn.Sequential(
        T.Normalize(mean=[195.333, 187.721, 185.050], std=[72.6773, 75.9105, 77.4861])
    ).to(device)
    for epoch in range(epochs):
        train_loss, train_metric = train(model=model, criterion=criterion, optimiser=optimizer,
                                         dataloader=train_dataloader, transforms=transforms_train,
                                         common=transforms_test)
        val_loss, val_metric = evaluate(model=model, criterion=criterion,
                                        dataloader=val_dataloader, transforms=transforms_test)
        wandb.log(
            {"train_loss": train_loss, "train_metric": train_metric, "val_loss": val_loss, "val_metric": val_metric})
        scheduler.step()
    torch.save(model.state_dict(), f"./model_libarary/model-{wandb.run.id}.pt")
    art = wandb.Artifact(f'mnist-nn-{wandb.run.id}', type="model")
    art.add_file(f"./model_libarary/model-{wandb.run.id}.pt", "model.pt")

    test_dataset = fashionDataset_1('test_tensor.pt')
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=100)
    index_1 = []
    for X in test_dataloader:
        X = X.to(device)
        X = transforms_test(X)
        y_hat = model(X)
        index = formal_output(y_hat)
        index_1.append(index)
    index = torch.cat(index_1, dim=0)
    index = index.cpu()
    index = index.numpy()
    np.savetxt(f"result/prediction_{wandb.run.id}.txt", index, fmt="%.d")
    art.add_file(f"result/prediction_{wandb.run.id}.txt", "prediction.output")
    wandb.log_artifact(art)
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
