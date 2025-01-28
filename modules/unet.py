import os
from typing import Sequence

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
        self.loss_function = DiceCELoss(include_background=True, to_onehot_y=True)
        self.lr = 3e-4

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch):
        image = batch["image"]
        label = batch["label"]
        prediction = self.model(image)
        loss = self.loss_function(prediction, label)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        image = batch["image"]
        label = batch["label"]
        output = self.model(image)
        loss = self.loss_function(output, label)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        prediction = torch.argmax(output, dim=1)
        # Log a gif and a point cloud
        for i in range(prediction.shape[0]):
            true_positives = (prediction[i] == label[i]).to(int) * label[i]
            incorrect = (prediction[i] != label[i]).to(int) * (prediction[i] + label[i])
            overlay = blend_images(image[i], true_positives, cmap="Greens")
            overlay = blend_images(overlay, incorrect, cmap="Reds")
            overlay = (overlay * 255).to(torch.uint8)
            wandb.log(
                {
                    f"Prediction {i}": wandb.Video(
                        overlay.permute(3, 0, 1, 2), fps=4, format="gif"
                    )
                }
            )
            vertices = make_point_cloud(
                array=prediction[i].cpu().numpy(),
                spacings=(1, 1, 1),
                color=(255, 0, 0),
            )
            wandb.log({f"Predictions {i}": wandb.Object3D(vertices)})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
