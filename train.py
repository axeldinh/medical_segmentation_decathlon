import lightning as L
import toml
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

import datasets
import modules

if torch.cuda.is_available():
    device = "cuda"
# elif torch.mps.is_available():
#     device = "mps"
else:
    device = "cpu"

if __name__ == "__main__":
    with open("config.toml", "r") as f:
        config = toml.load(f)

    model_cfg = config["model"]
    model = getattr(modules, model_cfg["name"])(**model_cfg["kwargs"])

    data_cfg = config["data"]
    data_module = getattr(datasets, data_cfg["name"])(**data_cfg["kwargs"])

    loggers = [
        TensorBoardLogger("logs"),
        WandbLogger(
            name="unet¨",
            project="medical_segmentation_decathlon",
            group="train",
            config=config,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=100,
        accelerator=device,
        logger=loggers,
        log_every_n_steps=1,
    )
    trainer.fit(model, data_module)
