import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import KMNIST49


def load_dataset(data_root, dataset_name, img_size, only_train=True, trans=None):
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]))
        test_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset_name == 'kmnist49':
        train_dataset = KMNIST49(
            root=data_root,
            train=True,
            download=True
        )
        test_dataset = KMNIST49(
            root=data_root,
            train=False,
            download=True
        )
    elif dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]),
            download=True
        )
    else:
        if not only_train:
            train_root = os.path.join(data_root, dataset_name, 'train')
        else:
            train_root = os.path.join(data_root, dataset_name)
        train_dataset = datasets.ImageFolder(
            # root=os.path.join(data_root, dataset_name, 'train'),
            root=train_root,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_root, dataset_name, 'test'),
            transform=trans) if not only_train else None
    return train_dataset, test_dataset


class DataLoader(object):
    def __init__(self, data_root, dataset_name, img_size, batch_size, with_label):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.with_label = with_label

    def get_loader(self, only_train=False, trans=None):
        train_dataset, test_dataset = load_dataset(
            self.data_root, self.dataset_name, self.img_size, only_train, trans)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4) if not only_train else None

        print(f'Total number of train data: {len(train_loader.dataset)}')
        if not only_train:
            print(f'Total number of test data: {len(test_loader.dataset)}')
        if self.with_label:
            print(f'Total number of classes: {len(train_dataset.classes)}\n')
        return train_loader, test_loader
