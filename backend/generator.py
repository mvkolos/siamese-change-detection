from torch.utils.data import DataLoader
from functools import wraps

@wraps(DataLoader)
def segmentation_generator(*args, **kwargs):
    while True:
        data_loader = DataLoader(*args, **kwargs)
        for batch in data_loader:
            images = [image.numpy() for image in batch['images']]
            labels = batch['mask']
            yield images, labels.numpy()