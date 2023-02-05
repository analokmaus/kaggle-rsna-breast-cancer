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
from pydicom.pixel_data_handlers import apply_windowing
from pydicom.pixel_data_handlers.util import apply_voi_lut


IMG_SIZE = 2048
WINDOW = False
VOI_LUT = True
VOI_LUT_SIGONLY = True
EXPORT_DIR = DATA_DIR/f'image_resized_{IMG_SIZE}{"V" if VOI_LUT else ""}{"S" if VOI_LUT_SIGONLY else ""}'
EXPORT_DIR.mkdir(exist_ok=True)
N_JOBS = 16


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
    # if dicom.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':  # ALREADY PROCESSED
    #     return

    img = dicom.pixel_array
    if VOI_LUT:
        if dicom.get('VOILUTFunction', None) != 'SIGMOID' and VOI_LUT_SIGONLY:
            img = img
        else:
            img = apply_voi_lut(img, dicom)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img
    img = (img - img.min()) / (img.max() - img.min())

    img = cv2.resize(img, (size, size))
    cv2.imwrite(str(EXPORT_DIR/f'{patient_id}/{image_id}.png'), (img * 255).astype(np.uint8))


train_images = list((DATA_DIR/'train_images/').glob('**/*.dcm'))
_ = ProgressParallel(n_jobs=N_JOBS)(
    delayed(process)(img_path, size=IMG_SIZE) for img_path in tqdm(train_images)
)
