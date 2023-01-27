import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import os
import pydicom
from general import DATA_DIR
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut


IMG_SIZE = 2048
VOI_LUT = True
VINDR_DIR = Path('input/rsna-breast-cancer-detection/vindr_mammo/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0')
EXPORT_DIR = DATA_DIR/f'vindr_mammo_resized_{IMG_SIZE}{"V" if VOI_LUT else ""}'
EXPORT_DIR.mkdir(exist_ok=True)
N_JOBS = 40


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def process(f, size=1024):
    patient_id = f.parent.name
    if not (EXPORT_DIR/patient_id).exists():
        (EXPORT_DIR/patient_id).mkdir(exist_ok=True)
    image_id = f.stem
    dicom = pydicom.dcmread(f)

    img = dicom.pixel_array
    if VOI_LUT:
        img = apply_voi_lut(img, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img
    img = (img - img.min()) / (img.max() - img.min())

    img = cv2.resize(img, (size, size))
    # cv2.imwrite(str(EXPORT_DIR/f'{patient_id}_{image_id}.png'), (img * 255).astype(np.uint8))
    cv2.imwrite(str(EXPORT_DIR/f'{patient_id}/{image_id}.png'), (img * 255).astype(np.uint8))



train_images = list((VINDR_DIR/'images/').glob('**/*.dicom'))
_ = ProgressParallel(n_jobs=N_JOBS)(
    delayed(process)(img_path, size=IMG_SIZE) for img_path in tqdm(train_images)
)
