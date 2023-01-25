import numpy as np
from pathlib import Path
import torch
import torch.utils.data as D
import cv2

from general import *


class PatientLevelDataset(D.Dataset):
    def __init__(
        self, df, image_dir, target_cols=['cancer'], aux_target_cols=[], 
        metadata_cols=[], sep='/', 
        preprocess=None, transforms=None, flip_lr=False, ddsm=False,
        # sampling strategy
        sample_num=1, view_category= [['MLO', 'LMO', 'LM', 'ML'], ['CC', 'AT']], replace=False, sample_criteria='high_value', 
        is_test=False, mixup_params=None, return_index=False):

        self.df = df
        if 'oversample_id' in df.columns:
            self.df_dict = {pid: pdf for pid, pdf in df.groupby(['oversample_id', 'patient_id', 'laterality'])}
        else:
            self.df_dict = {pid: pdf for pid, pdf in df.groupby(['patient_id', 'laterality'])}
        self.pids = list(self.df_dict.keys())
        self.image_dir = image_dir
        self.target_cols = target_cols
        self.aux_target_cols = aux_target_cols
        self.metadata_cols = metadata_cols
        self.preprocess = preprocess
        self.transforms = transforms
        self.flip_lr = flip_lr # Sorry this option is no longer 
        self.ddsm = ddsm
        self.is_test = is_test
        self.sample_num = sample_num
        self.view_category = view_category
        self.replace = replace
        self.sample_criteria = sample_criteria
        assert sample_criteria in ['high_value', 'low_value_for_implant']
        if mixup_params:
            assert 'alpha' in mixup_params.keys()
            self.mu = True
            self.mu_a = mixup_params['alpha']
        else:
            self.mu = False
        self.rt_idx = return_index
        self.sep = sep

    def __len__(self):
        return len(self.df_dict) # num_patients

    def _process_img(self, img):
        if self.preprocess:
            img = self.preprocess(image=img)['image']

        if self.transforms:
            if len(img.shape) == 4: # Tile x W x H x Ch
                output = []
                for tile in img:
                    output.append(self.transforms(image=tile)['image']) # -> torch
                output = torch.stack(output)
                img = output
            else: # W x H x Ch
                img = self.transforms(image=img)['image']
        
        return img

    def _load_image(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._process_img(img)
        return img

    def _load_best_image(self, df): # for test
        scores = []
        images = []
        iids = []
        if self.ddsm:
            is_implant = 0
        else:
            is_implant = df['implant'].values[0]
        for pid, iid in df[['patient_id', 'image_id']].values:
            if self.ddsm:
                img_path = self.image_dir/f'ddsm_{iid}.png'
            else:
                img_path = self.image_dir/f'{pid}{self.sep}{iid}.png'
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scores.append(img.mean())
            images.append(img)
            iids.append(iid)
        if is_implant and self.sample_criteria == 'low_value_for_implant':
            score_idx = np.argsort(scores)
        else:
            score_idx = np.argsort(scores)[::-1]
        output_imgs = [self._process_img(images[idx]) for idx in score_idx[:self.sample_num]]
        return output_imgs, [iids[idx] for idx in score_idx[:self.sample_num]]

    def _load_data(self, idx):
        pid = self.pids[idx]
        pdf = self.df_dict[pid]
        img = []
        img_ids = [] # replace?
        for iv, view_cat in enumerate(self.view_category):
            view0 = pdf.loc[pdf['view'].isin(view_cat) & ~pdf['image_id'].isin(img_ids)]
            if not self.replace and len(view0) == 0:
                view0 = pdf.loc[pdf['view'].isin(view_cat)]
            if len(view0) == 0:
                img0 = []
            else:
                if self.is_test:
                    img0, iid = self._load_best_image(view0)
                    if not self.replace:
                        img_ids.extend(iid)
                else:
                    view0 = view0.sample(min(self.sample_num, len(view0)))
                    img0 = []
                    for pid, iid in view0[['patient_id', 'image_id']].values:
                        if self.ddsm:
                            img_path = self.image_dir/f'ddsm_{iid}.png'
                        else:
                            img_path = self.image_dir/f'{pid}{self.sep}{iid}.png'
                        img0.append(self._load_image(img_path))
                        if not self.replace:
                            img_ids.append(iid)
            img.extend(img0)
        
        img = torch.stack(img, dim=0)
        expected_dim = self.sample_num * len(self.view_category)
        if img.shape[0] < expected_dim:
            img = torch.concat(
                [img, torch.zeros(
                    (expected_dim-img.shape[0], *img.shape[1:]), dtype=torch.float32)], dim=0)

        label = torch.from_numpy(pdf[self.target_cols+self.aux_target_cols].values[0].astype(np.float16))
        
        return img, label

    def __getitem__(self, idx):
        img, label = self._load_data(idx)

        if self.mu:
            idx2 = np.random.randint(0, len(self.images))
            lam = np.random.beta(self.mu_a, self.mu_a)
            img2, label2 = self._load_data(idx2)
            img = lam * img + (1 - lam) * img2
            label = lam * label + (1 - lam) * label2

        if self.rt_idx:
            return img, label, idx
        else:
            return img, label

    def get_labels(self):
        labels = []
        for idx in range(len(self.df_dict)):
            pid = self.pids[idx]
            pdf = self.df_dict[pid]
            labels.append(pdf[self.target_cols].values[0].reshape(1, 1).astype(np.float16))
        return np.concatenate(labels, axis=0)


class ImageLevelDataset(D.Dataset):
    def __init__(
        self, df, image_dir, target_cols=['cancer'], metadata_cols=[], sep='/', 
        preprocess=None, transforms=None,
        is_test=False, mixup_params=None, return_index=False, return_id=True):

        self.df = df
        self.image_dir = image_dir
        self.target_cols = target_cols
        self.metadata_cols = metadata_cols
        self.preprocess = preprocess
        self.transforms = transforms
        self.is_test = is_test
        if mixup_params:
            assert 'alpha' in mixup_params.keys()
            self.mu = True
            self.mu_a = mixup_params['alpha']
        else:
            self.mu = False
        self.rt_idx = return_index
        self.rt_id = return_id
        self.sep = sep

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.preprocess:
            img = self.preprocess(image=img)['image']

        if self.transforms:
            if len(img.shape) == 4: # Tile x W x H x Ch
                output = []
                for tile in img:
                    output.append(self.transforms(image=tile)['image']) # -> torch
                output = torch.stack(output)
                img = output
            else: # W x H x Ch
                img = self.transforms(image=img)['image']

        return img

    def _load_data(self, idx):
        pdf = self.df.iloc[idx]
        path0 = self.image_dir/f'{pdf.patient_id}{self.sep}{pdf.image_id}.png'
        img0 = self._load_image(path0) # (Ch x W x H) or (T x Ch x W x H)
        label = torch.tensor([pdf[self.target_cols].values[0]]).float()
        return img0, label

    def __getitem__(self, idx):
        pdf = self.df.iloc[idx]
        img, label = self._load_data(idx)
        patient_id = torch.tensor([pdf.patient_id])
        laterality = torch.tensor([int(pdf.laterality == 'R')])

        if self.mu:
            idx2 = np.random.randint(0, len(self.images))
            lam = np.random.beta(self.mu_a, self.mu_a)
            img2, label2 = self._load_data(idx2)
            img = lam * img + (1 - lam) * img2
            label = lam * label + (1 - lam) * label2

        output = [img, label]

        if self.rt_idx:
            output.append(idx)
        if self.rt_id:
            output.extend([patient_id, laterality])
        
        return tuple(output)

    def get_labels(self):
        return self.df[self.target_cols].values
