import glob
import os
import random

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms as tvt
from torchvision.io.image import ImageReadMode


class PkmCardSegmentationDataset(Dataset):
    def __init__(self, img_paths, label_paths, size=(512, 368), transform=None):
        self.images = img_paths
        self.labels = label_paths

        self.transform = transform
        if size is not None:
            self.resize = tvt.Resize(size, antialias=False)
        else:
            self.resize = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = read_image(self.images[idx], ImageReadMode.RGB)
        label = read_image(self.labels[idx], ImageReadMode.GRAY)

        if self.resize:
            image = self.resize(image)
            label = self.resize(label)

        image = (image / 255.).to(torch.float)
        label = (label != 0).to(torch.float)

        if self.transform is not None:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)
            label = (label != 0).to(torch.float)

        return image, label



class PkmCardSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 8, size=(512, 352), transform = None, use_noisy=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_noisy = use_noisy

        self.transform = transform

        self.size = size

        self.dt_train = None
        self.dt_valid = None

    def setup(self, stage: str = ""):
        images = glob.glob(os.path.join(self.data_dir, "images") + "/*.jpg")
        labels = glob.glob(os.path.join(self.data_dir, "masks") + "/*.jpg")

        if self.use_noisy:
            images += glob.glob(os.path.join(self.data_dir, "noisy/images") + "/*.jpg")
            labels += glob.glob(os.path.join(self.data_dir, "noisy/asks") + "/*.jpg")

            # shuffle to have reasonable validation splits that do not contain only
            # noisy labels
            temp = list(zip(images, labels))
            random.shuffle(temp)
            images, labels = zip(*temp)

        split = int(len(images)  * .8)

        self.dt_train = PkmCardSegmentationDataset(images[:split], labels[:split], size=self.size, transform=self.transform)
        self.dt_valid = PkmCardSegmentationDataset(images[split:], labels[split:], size=self.size)


    def train_dataloader(self):
        return DataLoader(self.dt_train, batch_size=self.batch_size, pin_memory=True, num_workers=int(os.cpu_count()/2), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dt_valid, batch_size=self.batch_size, pin_memory=True, num_workers=int(os.cpu_count()/2))
