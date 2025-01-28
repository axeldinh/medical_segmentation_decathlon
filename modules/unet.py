from functools import reduce
from typing import Dict, Sequence

import torch
import wandb
from lightning import LightningModule
from monai.losses import DiceCELoss
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import UNet
from monai.visualize.utils import blend_images

from utils.to_obj import make_point_cloud


class UNetModule(LightningModule):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        optimizer_config: Dict = {"name": "Adam", "kwargs": {"lr": 3e-4}},
    ):
        super().__init__()
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )
        self.loss_function = DiceCELoss(include_background=False, to_onehot_y=True)
        self.optimizer_config = optimizer_config

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch):
        image = batch["image"]
        label = batch["label"]
        prediction = self.model(image)
        loss = self.loss_function(prediction, label)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch):
        image = batch["image"]
        label = batch["label"]
        output = self.model(image)
        loss = self.loss_function(output, label)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        prediction = torch.argmax(output, dim=1)
        self.validation_step_outputs.append(
            {"prediction": prediction, "label": label, "image": image}
        )

        return prediction

    def on_validation_epoch_end(self):
        predictions = [item["prediction"] for item in self.validation_step_outputs]
        labels = [item["label"] for item in self.validation_step_outputs]
        images = [item["image"] for item in self.validation_step_outputs]

        predictions = reduce(lambda x, y: x + y, [[x for x in b] for b in predictions])
        labels = reduce(lambda x, y: x + y, [[x for x in b] for b in labels])
        images = reduce(lambda x, y: x + y, [[x for x in b] for b in images])

        for i in range(len(predictions)):
            pred = predictions[i]
            label = labels[i]
            img = images[i]
            true_positives = (pred == label).to(int) * label
            incorrect = (pred != label).to(int) * (pred + label)
            overlay = blend_images(img, true_positives, cmap="Greens")
            overlay = blend_images(overlay, incorrect, cmap="Reds")
            overlay = (overlay * 255).to(torch.uint8)
            wandb.log(
                {
                    f"Gif Validation {i}": wandb.Video(
                        overlay.permute(3, 0, 1, 2).detach().cpu(), fps=4, format="gif"
                    )
                }
            )
            vertices = make_point_cloud(
                array=pred.cpu().numpy(),
                spacings=(1, 1, 1),
                color=(255, 0, 0),
            )
            wandb.log({f"Point Cloud Validation {i}": wandb.Object3D(vertices)})

    def configure_optimizers(self):
        return getattr(torch.optim, self.optimizer_config["name"])(
            self.model.parameters(), **self.optimizer_config["kwargs"]
        )
