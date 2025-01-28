import os

import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from monai.apps.datasets import DecathlonDataset
from monai.data import DataLoader

from .transforms import make_transforms


class HeartDataModule(LightningDataModule):
    def __init__(self, root="data", transforms_kwargs={}, batch_size=1, val_split=0.1, num_workers=4):
        super().__init__()
        self.root = root
        self.batch_size = 1
        self.num_workers = num_workers
        self.transforms_kwargs = transforms_kwargs
        self.data_kwargs = dict(
            root_dir=self.root,
            section="training",
            task="Task02_Heart",
            val_frac=val_split,
            download=True,
        )

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.root, "Task02_Heart")):
            DecathlonDataset(**self.data_kwargs)

    def setup(self, stage):
        transforms = make_transforms(self.transforms_kwargs)

        train_transform = transforms[0]
        val_transform = transforms[1]
        test_transform = transforms[2]

        self.data_kwargs["download"] = False
        if stage == "fit":
            self.data_kwargs["section"] = "training"
            self.train = DecathlonDataset(transform=train_transform, **self.data_kwargs)
            self.data_kwargs["section"] = "validation"
            self.val = DecathlonDataset(transform=val_transform, **self.data_kwargs)

        if stage == "predict":
            self.data_kwargs["section"] = "test"
            self.test = DecathlonDataset(transform=test_transform, **self.data_kwargs)

    def collate_fn(self, batch):
        sizes = [item["image"].shape[-1] for item in batch]
        max_size = max(sizes)

        padded_images = []
        padded_labels = []

        for item in batch:
            image = item["image"]
            label = item["label"]
            padded_image = F.pad(
                image, (0, max_size - image.shape[-1]), mode="constant", value=0
            )
            padded_label = F.pad(
                label, (0, max_size - label.shape[-1]), mode="constant", value=0
            )
            padded_images.append(padded_image)
            padded_labels.append(padded_label)

        padded_batch = {
            "image": torch.stack(padded_images),
            "label": torch.stack(padded_labels),
        }

        return padded_batch

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers,
        )
