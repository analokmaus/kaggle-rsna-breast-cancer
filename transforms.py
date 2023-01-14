import numpy as np
import random
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform


'''
Preprocessing
'''
def get_blue_ratio(img):
    # (N, C, W, H) image
    rgbs = img.transpose(0, 3, 1, 2).mean(2).mean(2)  # N, C
    br = (100 + rgbs[:, 2]) * 256 / \
        (1 + rgbs[:, 0] + rgbs[:, 1]) / (1 + rgbs.sum(1))
    return br


def make_tiles(img, sz=128, num_tiles=4, criterion='darkness', concat=True, dropout=0.0):
    if len(img.shape) == 2:
        img = img[:, :, None]
    w, h, ch = img.shape
    pad0, pad1 = (sz - w % sz) % sz, (sz - h % sz) % sz
    padding = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
    img = np.pad(img, padding, mode='constant', constant_values=0)
    img = img.reshape(img.shape[0]//sz, sz, img.shape[1]//sz, sz, ch)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, ch)
    valid_count = len(img)
    if len(img) < num_tiles:
        padding = [[0, num_tiles-len(img)], [0, 0], [0, 0], [0, 0]]
        img = np.pad(img, padding, mode='constant', constant_values=255)
    if criterion == 'darkness':
        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_tiles]
    elif criterion == 'blue-ratio':
        idxs = np.argsort(get_blue_ratio(img) * -1)[:num_tiles]
    elif criterion == 'brightness':
        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[::-1][:num_tiles]
    else:
        raise ValueError(criterion)
    if concat:
        tile_row = int(np.sqrt(num_tiles))
        img = cv2.hconcat(
            [cv2.vconcat([_img for _img in img[idxs[i:i+tile_row]]]) \
                for i in np.arange(0, num_tiles, tile_row)])
    else:
        img = img[idxs]
        if dropout > 0:
            valid_count = min(valid_count, num_tiles)
            drop_count = round(valid_count * dropout)
            if drop_count > 0:
                drop_index = random.sample(range(valid_count), drop_count)
                img[drop_index] = img[drop_index].mean()
    return img


class ImageToTile(ImageOnlyTransform):

    def __init__(self, tile_size=256, num_tiles=25, concat=True, 
                 criterion='darkness', dropout=0.0,
                 always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.concat = concat
        self.criterion = criterion
        self.dropout = dropout

    def apply(self, img, **params):
        return make_tiles(
            img, self.tile_size, self.num_tiles, self.criterion, self.concat, self.dropout)
    
    def get_transform_init_args_names(self):
        return ('tile_size', 'num_tiles', 'concat', 'criterion', 'dropout')
    


class CropROI(ImageOnlyTransform):

    def __init__(self, threshold=0.1, buffer=30, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.threshold = threshold
        self.buffer = buffer

    def crop_roi(self, img, threshold=0.1, buffer=50):
        y_max, x_max = img.shape
        img2 = img > img.mean()
        y_mean = img2.mean(0)
        x_mean = img2.mean(1)
        x_mean[:buffer] = 0
        x_mean[-buffer:] = 0
        y_mean[:buffer] = 0
        y_mean[-buffer:] = 0
        y_mean = (y_mean - y_mean.min() + 1e-4) / (y_mean.max() - y_mean.min() + 1e-4)
        x_mean = (x_mean - x_mean.min() + 1e-4) / (x_mean.max() - x_mean.min() + 1e-4)
        y_slice = np.where(y_mean > threshold)[0]
        x_slice = np.where(x_mean > threshold)[0]
        if len(x_slice) == 0:
            x_start, x_end = 0, x_max
        else:
            x_start, x_end = max(x_slice.min() - buffer, 0), min(x_slice.max() + buffer, x_max)
        if len(y_slice) == 0:
            y_start, y_end = 0, y_max
        else:
            y_start, y_end = max(y_slice.min() - buffer, 0), min(y_slice.max() + buffer, y_max)
        return img[x_start:x_end, y_start:y_end]

    def apply(self, img, **params):
        return self.crop_roi(img, self.threshold, self.buffer)
    
    def get_transform_init_args_names(self):
        return ('threshold', 'buffer')


class AutoFlip(ImageOnlyTransform):

    def __init__(self, sample_width=100, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.sample_width = sample_width

    def apply(self, img, **params):
        if img[:, :self.sample_width].sum() <= img[:, -self.sample_width:].sum():
            img = img[:, ::-1]
        return img
    
    def get_transform_init_args_names(self):
        return ('sample_width')
    