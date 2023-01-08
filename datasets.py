import numpy as np
from pathlib import Path
import torch
import torch.utils.data as D
import cv2

from general import *


class PatientLevelDataset(D.Dataset):
    def __init__(
        self, df, image_dir, target_cols=['cancer'],
        preprocess=None, transforms=None, 
        is_test=False, mixup_params=None, return_index=False):

        self.df = df
        self.df_dict = {pid: pdf for pid, pdf in df.groupby(['patient_id', 'laterality'])}
        self.pids = list(self.df_dict.keys())
        self.image_dir = image_dir
        self.target_cols = target_cols
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        self.view_category = [['MLO', 'LMO', 'LM', 'ML'], ['CC', 'AT']]
        if mixup_params:
            assert 'alpha' in mixup_params.keys()
            self.mu = True
            self.mu_a = mixup_params['alpha']
        else:
            self.mu = False
        self.rt_idx = return_index

    def __len__(self):
        return len(self.df_dict) # num_patients

    def _load_image(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.preprocess:
            img = self.preprocess(image=img)['image']

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img

    def _load_data(self, idx):
        pid = self.pids[idx]
        pdf = self.df_dict[pid]
        view0 = pdf.query('view.isin(@self.view_category[0])')
        if len(view0) == 0:
            img0 = None
        else:
            view0 = view0.sample().iloc[0]
            path0 = self.image_dir/f'{view0.patient_id}/{view0.image_id}.png'
            img0 = self._load_image(path0)

        view1 = pdf.query('view.isin(@self.view_category[1])')
        if len(view1) == 0:
            img1 = None
        else:
            view1 = view1.sample().iloc[0]
            path1 = self.image_dir/f'{view1.patient_id}/{view1.image_id}.png'
            img1 = self._load_image(path1)

        if img0 is None and not img1 is None:
            img0 = torch.zeros_like(img1)
        elif img1 is None and not img0 is None:
            img1 = torch.zeros_like(img0)

        if pid[1] == 'R':
            img0 = torch.flip(img0, dims=(2,))
            img1 = torch.flip(img1, dims=(2,))

        label = torch.from_numpy(view0[self.target_cols].values.astype(np.float16))
        return img0, img1, label

    def __getitem__(self, idx):
        img0, img1, label = self._load_data(idx)

        if self.mu:
            idx2 = np.random.randint(0, len(self.images))
            lam = np.random.beta(self.mu_a, self.mu_a)
            img01, img11, label1 = self._load_data(idx2)
            img0 = lam * img0 + (1 - lam) * img01
            img1 = lam * img1 + (1 - lam) * img11
            label = lam * label + (1 - lam) * label1

        if self.rt_idx:
            return img0, img1, label, idx
        else:
            return img0, img1, label
