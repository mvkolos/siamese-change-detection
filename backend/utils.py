import numpy as np
from aeronet.dataset import BandCollection


def to_x32(x):
    return ((x - 1) // 32 + 1) * 32


def pad_x32(img):
    w, h, c = img.shape

    h_x32 = to_x32(h)
    w_x32 = to_x32(w)

    res = np.zeros((w_x32, h_x32, c))

    for i in range(c):
        res[:, :, i] = np.pad(img[:, :, i], ((0, w_x32 - w), (0, h_x32 - h)), mode='reflect')

    return res, (w_x32 - w, h_x32 - h)


def load_bc(data_dir, element_name, channels, output_labels):
    return BandCollection([f'{data_dir}/{element_name}_channel_{ch}.tif' for ch in channels] + \
                          [f'{data_dir}/{element_name}_class_{cls}.tif' for cls in output_labels])
