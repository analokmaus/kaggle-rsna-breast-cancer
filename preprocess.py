from functools import partial
import numpy as np
import pandas as pd
import cv2
import skimage.io as io
import pyvips
import argparse
from pprint import pprint
from multiprocessing import Pool
import gc

from general import *
from utils import notify_me


def get_metadata(img_paths):
    metadata = []
    bin_thres = 200
    for i, img_p in enumerate(image_paths):
        iid = img_p.stem
        # img = cv2.imread(str(img_p), 0)
        img = io.imread(str(img_p))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, img_bin = cv2.threshold(img_gray, bin_thres, 255, cv2.THRESH_BINARY)
        img_size = img.shape
        metadata.append({
            'image_id': iid,
            'img_height': img_size[0],
            'img_width': img_size[1],
            'ROI_ratio': (img_bin == 0).sum() / (img_size[0]*img_size[1])
        })
    metadata = pd.DataFrame(metadata)
    return metadata


def preprocess(img_path, resize_factor, interpolate, export_path):
    iid = img_path.stem
    img = pyvips.Image.new_from_file(str(img_path))
    img = img.resize(1/resize_factor, kernel=interpolate)
    img.write_to_file(str(export_path/f'{iid}.png'))
    del img; gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", action='store_true')
    parser.add_argument("--resizefactor", type=int, default=16)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--interpolate", type=str, default='linear')
    parser.add_argument("--export_path", type=str, default=None)
    opt = parser.parse_args()
    pprint(opt)

    image_paths = list(DATA_DIR.glob('**/*.tif'))
    if opt.export_path is None:
        export_path = Path(f'input/resized/{opt.resizefactor}/')
    else:
        export_path = Path(opt.export_path).expanduser()
    print(len(image_paths))
    export_path.mkdir(exist_ok=True, parents=True)

    if opt.metadata:
        metadata = get_metadata(image_paths)
        metadata.to_csv('input/metadata.csv', index=False)
        import sys; sys.exit()
    
    
    if opt.n_jobs == 1:
        for i, img_p in enumerate(image_paths):
            # img = io.imread(str(img_p))
            # img = cv2.resize(img, dsize=None, fx=1/opt.resizefactor, fy=1/opt.resizefactor)
            # cv2.imwrite(str(export_path/f'{iid}.png'), img)
            preprocess(img_p, opt.resizefactor, opt.interpolate, export_path)
    else:
        with Pool(opt.n_jobs) as p:
            res = p.map(partial(
                preprocess, 
                resize_factor=opt.resizefactor, 
                interpolate=opt.interpolate, 
                export_path=export_path), image_paths)

    notify_me(f'preprocess.py {opt.resizefactor}/{opt.interpolate} finished')
