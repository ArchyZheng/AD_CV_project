# This is dataset source code
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import pprint
from torchvision.io import read_image
import torch.nn as nn
from torchvision import transforms as T


transform = nn.Sequential(
    T.Resize((224, 224))
)


class fashionDataset(Dataset):
    def __init__(self, data_file: str, picture_list_dir: str, label_list_dir: str):
        self.dir_list = [data_file, picture_list_dir, label_list_dir]
        self.picture_list = pd.read_csv(self.dir_list[0] + '/' + self.dir_list[1])
        self.label_list = pd.read_csv(self.dir_list[0] + '/' + self.dir_list[2], sep=" ")
        self.picture_dir_name = self.dir_list[0]

    def __getitem__(self, index):
        img_path = self.picture_dir_name + '/' + str(self.picture_list.values[index][0])
        img_tensor = read_image(img_path).float()
        # the operation of resize must in the dataset, cause the input of different picture has different size
        img_tensor = transform(img_tensor)
        # The classic label of multilabel task
        # TODO: try other operation such like separate the output into six sub-field.
        index = self.label_list.values[index] + [0, 7, 10, 13, 17, 23]
        label = torch.zeros(26)
        label[index] = 1
        return img_tensor, label.type(torch.LongTensor)

    def __len__(self):
        return len(self.picture_list)


def get_train_or_val_dataloader(data_file: str, picture_list_dir: str, label_list_dir: str, shuffle: bool, batch_size: int):
    '''
    :param data_file: the location of dataset "main fold"
    :param picture_lis_dir: the location of train.txt/val.txt/test.txt
    :param label_list_dir: the location of train_attr.txt/val_attr.txt
    :param shuffle: is shuffle?
    :param batch_size:
    :return: the object of dataloder class
    >> dataloader = get_train_dataloader(data_file='./FashionDataset', picture_list_dir='split/train.txt',
                    label_list_dir='split/train_attr.txt', shuffle=True, batch_size=16)
    >> len(next(iter(dataloader))) == 16
    True
    '''
    dataset = fashionDataset(data_file=data_file, picture_list_dir=picture_list_dir, label_list_dir=label_list_dir)
    return DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size)