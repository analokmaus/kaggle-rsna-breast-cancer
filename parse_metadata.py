import numpy as np
import pandas as pd
from pathlib import Path
import pydicom
from tqdm.auto import tqdm


TRAIN = pd.read_csv('input/rsna-breast-cancer-detection/train.csv')
DICOM_DIR = Path('input/rsna-breast-cancer-detection/train_images')
N_JOBS = 8


metadata = []
for pid, iid in tqdm(TRAIN[['patient_id', 'image_id']].values):
    dicom = pydicom.dcmread(str(DICOM_DIR/f'{pid}/{iid}.dcm'))
    try:
        metadata.append({
            'patient_id': pid,
            'image_id': iid,
            'content_date': int(dicom.ContentDate),
            'content_time': float(dicom.ContentTime)
        })
    except:
        print(f'Failled: {pid}/{iid}')
metadata = pd.DataFrame(metadata)


TRAIN = TRAIN.merge(metadata, on=['patient_id', 'image_id'], how='left')
TRAIN.to_csv('input/rsna-breast-cancer-detection/train_meta.csv', index=False)