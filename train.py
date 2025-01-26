import lightning as L
import lightning.pytorch.loggers as lightning_loggers
import toml
import torch

import datasets
import modules

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__ == "__main__":
    with open("config.toml", "r") as f:
        config = toml.load(f)

    model_cfg = config["model"]
    model = getattr(modules, model_cfg["name"])(**model_cfg["kwargs"])

    data_cfg = config["data"]
    data_module = getattr(datasets, data_cfg["name"])(**data_cfg["kwargs"])

    loggers = []
    for name in config["loggers"]["names"]:
        kwargs = config["loggers"][name].copy()
        if name == "WandbLogger":
            kwargs["config"] = config
        loggers.append(getattr(lightning_loggers, name)(**kwargs))

    trainer = L.Trainer(logger=loggers, **config["trainer"])
    trainer.fit(model, data_module)
