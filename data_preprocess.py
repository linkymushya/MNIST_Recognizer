from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


def data_loader_MNIST():
    trans = transforms.ToTensor()
    train_datasets = torchvision.datasets.MNIST(root='./data',
                                                train=True, transform=trans, download=True)
    test_datasets = torchvision.datasets.MNIST(root='./data',
                                               train=False, transform=trans, download=True)

    train_dataloader = DataLoader(train_datasets, batch_size=16, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_datasets, batch_size=16, drop_last=True, shuffle=True)

    return train_dataloader, test_dataloader
