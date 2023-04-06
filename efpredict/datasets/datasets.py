from torch.utils.data import Dataset
import torch
from torchvision.transforms.functional import to_tensor

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

        # Process each frame individually and convert to tensor
        tensor_frames = [to_tensor(frame) for frame in data]

        # Stack the frames along a new dimension
        data = torch.stack(tensor_frames, dim=0)
        data = data.permute(1, 0, 2, 3).unsqueeze(0)
        return data, torch.tensor(-1)  # -1 indicates unlabelled data
