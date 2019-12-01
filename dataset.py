import os

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


class KMNIST49(torch.utils.data.Dataset):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz',
    ]

    def __init__(self, root, train=True, download=False):
        super(KMNIST49, self).__init__()

        self.data_root = os.path.join(root, 'KMNIST-49')

        if download:
            self._download()

        if train:
            img_path = os.path.join(self.data_root, 'k49-train-imgs.npz')
            label_path = os.path.join(self.data_root, 'k49-train-labels.npz')
        else:
            img_path = os.path.join(self.data_root, 'k49-test-imgs.npz')
            label_path = os.path.join(self.data_root, 'k49-test-labels.npz')

        self.data = np.load(img_path)['arr_0']
        self.targets = np.load(label_path)['arr_0']
        self.classes = list(range(49))

        data_mean = self.data.mean() / 255.0
        data_std = self.data.std() / 255.0

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            # transforms.Normalize((data_mean,), (data_std,))
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.expand_dims(img, axis=-1)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _download(self):
        os.makedirs(self.data_root, exist_ok=True)
        for url in self.urls:
            filename = url.rpartition('/')[-1]
            torchvision.datasets.utils.download_url(
                url, root=self.data_root, filename=filename, md5=None)
