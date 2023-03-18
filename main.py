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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16 # TODO: changed by wandb
    train_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/train.txt',
                                                   label_list_dir='split/train_attr.txt', shuffle=True, batch_size=batch_size)

    val_dataloader = get_train_or_val_dataloader(data_file='./FashionDataset', picture_list_dir='split/val.txt',
                                                   label_list_dir='split/val_attr.txt', shuffle=True, batch_size=batch_size)
    transform = nn.Sequential(
        T.Resize((224, 224))
    )
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = baseResnet()
    model.to(device)
    epochs = 200 # TODO: changed by wandb
    # TODO: optimizer dict
    optimizer = SGD(params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    # TODO: scheduler dict
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)


    criterion = nn.MultiLabelMarginLoss().to(device)
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0

        val_loss = 0
        val_acc = 0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            y.to(device)
            model.train()
            input = []
            for i in range(len(X)):
                img_tensor = read_image(X[i]).float().to(device)
                img_tensor = transform(img_tensor)
                input.append(img_tensor)
            input = torch.cat(input, 0).reshape(-1, 3, 224, 224)
            y_hat = model(input)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            train_loss += batch_size * loss.item()

        # for X, Y in val_dataloader:
        #     model.eval()
        #     input = []
        #     for i in range(len(X)):
        #         img_tensor = read_image(X[i]).float().to(device)
        #         img_tensor = transform(img_tensor)
        #         input.append(img_tensor)
        #
        # break
        #     # TODO: get loss

if __name__ == "__main__":
    main()
