import monai.transforms as mT

def make_transforms(transforms_info):
    """Make a transform from the given transforms info."""
    transforms = ()

    for split in transforms_info.keys():
        transforms_names = transforms_info[split]["names"]
        transforms_kwargs = [transforms_info[split].get(name, {}) for name in transforms_names]
        transform = mT.Compose([
            getattr(mT, name)(**kwargs) for name, kwargs in zip(transforms_names, transforms_kwargs)
        ])
        transforms += (transform,)


    return transforms
