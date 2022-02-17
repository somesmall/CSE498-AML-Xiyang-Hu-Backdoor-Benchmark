import copy
import numpy as np

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class MixMatchDataset(Dataset):
    """Semi-supervised MixMatch dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        semi_idx (np.array): An 0/1 (labeled/unlabeled) array with
            shape `(len(dataset), )`.
        labeled (bool, optional): If True, creates dataset from labeled set, otherwise
            creates from unlabeled set (default: True).
    """

    def __init__(self, dataset, semi_idx, labeled=True):
        super(MixMatchDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        if labeled:
            self.semi_indice = np.nonzero(semi_idx == 1)[0]
        else:
            self.semi_indice = np.nonzero(semi_idx == 0)[0]
        self.labeled = labeled
        self.prefetch = self.dataset.prefetch
        self.mean, self.std = self.dataset.mean, self.dataset.std

    def __getitem__(self, index):
        if self.labeled:
            item = self.dataset[self.semi_indice[index]]
            item["labeled"] = True
        else:
            item1 = self.dataset[self.semi_indice[index]]
            item2 = self.dataset[self.semi_indice[index]]
            img1, img2 = item1.pop("img"), item2.pop("img")
            item1.update({"img1": img1, "img2": img2})
            item = item1
            item["labeled"] = False

        return item

    def __len__(self):
        return len(self.semi_indice)


class SelfPoisonDataset(Dataset):
   

    def __init__(self, dataset, transform):
        super(SelfPoisonDataset, self).__init__()
        self.dataset = copy.deepcopy(dataset)
        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.poison_idx = self.dataset.poison_idx
        self.bd_transform = self.dataset.bd_transform
        self.target_label = self.dataset.target_label

        self.pre_transform = transform["pre"]
        self.primary_transform = transform["primary"]
        self.remaining_transform = self.dataset.remaining_transform
        self.prefetch = self.dataset.prefetch
        if self.prefetch:
            self.mean, self.std = self.dataset.mean, self.dataset.std

    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            with open(self.data[index], "rb") as f:
                img = np.array(Image.open(f).convert("RGB"))
        else:
            img = self.data[index]
        target = self.targets[index]
        poison = 0
        origin = target  # original target
        if self.poison_idx[index] == 1:
            img1 = self.bd_first_augment(img, bd_transform=self.bd_transform)
            img2 = self.bd_first_augment(img, bd_transform=self.bd_transform)
            target = self.target_label
            poison = 1
        else:
            img1 = self.bd_first_augment(img, bd_transform=None)
            img2 = self.bd_first_augment(img, bd_transform=None)
        item = {
            "img1": img1,
            "img2": img2,
            "target": target,
            "poison": poison,
            "origin": origin,
        }

        return item

    def __len__(self):
        return len(self.data)

    def bd_first_augment(self, img, bd_transform=None):
        # Pre-processing transformations (HWC ndarray->HWC ndarray).
        img = Image.fromarray(img)
        img = self.pre_transform(img)
        img = np.array(img)
        # Backdoor transformationss (HWC ndarray->HWC ndarray).
        if bd_transform is not None:
            img = bd_transform(img)
        # Primary and the remaining transformations (HWC ndarray->CHW tensor).
        img = Image.fromarray(img)
        img = self.primary_transform(img)
        img = self.remaining_transform(img)

        if self.prefetch:
            # HWC ndarray->CHW tensor with C=3.
            img = np.rollaxis(np.array(img, dtype=np.uint8), 2)
            img = torch.from_numpy(img)

        return img
