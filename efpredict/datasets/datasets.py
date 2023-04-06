from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor

class LabelledDataset(Dataset):
    def __init__(self, labelled_data, transform=None):
        self.labelled_data = labelled_data
        self.transform = transform

    def __len__(self):
        return len(self.labelled_data)

    def __getitem__(self, idx):
        data, label = self.labelled_data[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

class UnlabelledDataset(Dataset):
    def __init__(self, unlabelled_data, transform=None):
        self.unlabelled_data = unlabelled_data
        self.transform = transform

    def __len__(self):
        return len(self.unlabelled_data)

    def __getitem__(self, idx):
        data = self.unlabelled_data[idx]
        if self.transform:
            data = self.transform(data)
        data = ToTensor()(data)  # Convert data to tensor
        return data, torch.tensor(-1)  # -1 indicates unlabelled data
