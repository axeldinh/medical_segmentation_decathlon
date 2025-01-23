import monai.transforms as mT

def make_transforms(transforms_info):
    """Make a transform from the given transforms info."""
    transforms = ()

    for split in transforms_info.keys():
        train_transform = mT.Compose([
            getattr(mT, t["name"])(**(t["kwargs"] if t["kwargs"] else {}))
            for t in transforms_info[split]
        ])
        transforms += (train_transform,)


    return transforms
