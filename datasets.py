import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.utils.data as D
import cv2

from general import *
from transforms import AutoFlip


class PatientLevelDataset(D.Dataset):
    def __init__(
        self, df, image_dir, target_cols=['cancer'], aux_target_cols=[], 
        metadata_cols=[], sep='/', bbox_path=None, 
        preprocess=None, transforms=None, flip_lr=False,
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
        if bbox_path is None:
            self.bbox = None
        else:
            self.bbox = pd.read_csv(bbox_path).set_index('name').to_dict(orient='index')
        self.preprocess = preprocess
        self.transforms = transforms
        self.flip_lr = flip_lr # Sorry this option is no longer
        self.is_test = is_test
        self.sample_num = sample_num
        self.view_category = view_category
        self.replace = replace
        self.sample_criteria = sample_criteria
        assert sample_criteria in ['high_value', 'low_value_for_implant', 'latest', 'valid_area']
        if mixup_params:
            assert 'alpha' in mixup_params.keys()
            self.mu = True
            self.mu_a = mixup_params['alpha']
        else:
            self.mu = False
        self.rt_idx = return_index
        self.sep = sep

    def update_df(self, new_df):
        self.df = pd.concat([self.df, new_df])
        if 'oversample_id' in self.df.columns:
            self.df_dict = {pid: pdf for pid, pdf in self.df.groupby(['oversample_id', 'patient_id', 'laterality'])}
        else:
            self.df_dict = {pid: pdf for pid, pdf in self.df.groupby(['patient_id', 'laterality'])}
        self.pids = list(self.df_dict.keys())

    def __len__(self):
        return len(self.df_dict) # num_patients

    def _process_img(self, img, bbox=None):
        if self.preprocess:
            if bbox is None:
                img = self.preprocess(image=img)['image']
            else:
                img_h, img_w = img.shape
                bbox[2] = min(bbox[2], img_h)
                bbox[3] = min(bbox[3], img_w)
                img = self.preprocess(image=img, bboxes=[bbox])['image']

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

    def _load_image(self, path, bbox=None):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._process_img(img, bbox=bbox)
        return img

    def _get_file_path(self, patient_id, image_id):
        return self.image_dir/f'{patient_id}{self.sep}{image_id}.png'

    def _get_bbox(self, patient_id, image_id):
        if self.bbox is not None:
            key = f'{patient_id}/{image_id}.png'
            if key in self.bbox.keys():
                bbox = self.bbox[key]
                bbox = [bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax'], 'YOLO']
            else:
                bbox = [0, 0, 100000, 100000, 'YOLO'] # dummy
        else:
            bbox = None
        return bbox

    def _load_best_image(self, df): # for test
        if self.sample_criteria == 'latest':
            try:
                latest_idx = np.argmax(df['content_date'])
                df2 = df.iloc[latest_idx:latest_idx+1]
            except:
                pass
        else:
            df2 = df
        scores = []
        images = []
        bboxes = []
        iids = []
        if 'implant' in df2.columns:
            is_implant = df2['implant'].values[0]
        else:
            is_implant = 0
        for pid, iid in df2[['patient_id', 'image_id']].values:
            img_path = self._get_file_path(pid, iid)
            bbox = self._get_bbox(pid, iid)
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.sample_criteria == 'high_value':
                score = img.mean()
            elif self.sample_criteria == 'low_value_for_implant':
                score = img.mean()
                if is_implant:
                    score = 1 / (score + 1e-4)
            elif self.sample_criteria == 'valid_area':
                score = ((16 < img) & (img < 160)).mean()
                if is_implant:
                    score = 1 / (score + 1e-4)
            bboxes.append(bbox)
            scores.append(score)
            images.append(img)
            iids.append(iid)
        score_idx = np.argsort(scores)[::-1]
        # print(f'{pid}/{[iids[idx] for idx in score_idx[:self.sample_num]][0]}.png')
        output_imgs = [self._process_img(images[idx], bboxes[idx]) for idx in score_idx[:self.sample_num]]
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
                        img_path = self._get_file_path(pid, iid)
                        bbox = self._get_bbox(pid, iid)
                        img0.append(self._load_image(img_path, bbox))
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


class PatientLevelDatasetLR(D.Dataset):
    '''
    Output: LRLRLR images, target
    '''
    def __init__(
        self, df, image_dir, target_cols=['cancer'], aux_target_cols=[], 
        metadata_cols=[], sep='/', bbox_path=None, 
        preprocess=None, transforms=None, flip_lr=False, 
        img_size=2048, transform_imagewise=False, separate_channel=True,
        # sampling strategy
        view_category= [['MLO', 'LMO', 'LM', 'ML'], ['CC', 'AT']], sample_criteria='high_value', 
        is_test=False, return_index=False):

        self.df = df
        if 'oversample_id' in df.columns:
            self.df_dict = {pid: pdf for pid, pdf in df.groupby(['oversample_id', 'patient_id'])}
        else:
            self.df_dict = {pid: pdf for pid, pdf in df.groupby('patient_id')}
        self.pids = list(self.df_dict.keys())
        self.image_dir = image_dir
        self.target_cols = target_cols
        self.aux_target_cols = aux_target_cols
        self.metadata_cols = metadata_cols
        if bbox_path is None:
            self.bbox = None
        else:
            self.bbox = pd.read_csv(bbox_path).set_index('name').to_dict(orient='index')
        self.preprocess = preprocess
        self.transforms = transforms
        self.img_size = img_size
        self.transform_imagewise = transform_imagewise
        self.separate_channel = separate_channel
        self.flip_lr = flip_lr
        self.flip_t = AutoFlip(200)
        self.is_test = is_test
        self.view_category = view_category
        self.sample_criteria = sample_criteria
        assert sample_criteria in ['high_value', 'low_value_for_implant']
        self.rt_idx = return_index
        self.sep = sep

    def update_df(self, new_df):
        self.df = pd.concat([self.df, new_df])
        if 'oversample_id' in self.df.columns:
            self.df_dict = {pid: pdf for pid, pdf in self.df.groupby(['oversample_id', 'patient_id'])}
        else:
            self.df_dict = {pid: pdf for pid, pdf in self.df.groupby('patient_id')}
        self.pids = list(self.df_dict.keys())

    def __len__(self):
        return len(self.df_dict) # num_patients

    def _preprocess_img(self, img, bbox=None):
        if self.preprocess:
            if bbox is None:
                img = self.preprocess(image=img)['image']
            else:
                img_h, img_w = img.shape
                bbox[2] = min(bbox[2], img_h)
                bbox[3] = min(bbox[3], img_w)
                img = self.preprocess(image=img, bboxes=[bbox])['image']

        return img
    
    def _augment_img(self, img):
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        return img

    def _load_image(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.flip_lr:
            img = self.flip_t(image=img)['image']
        return img[:, :, None]
        
    def _get_file_path(self, patient_id, image_id):
        return self.image_dir/f'{patient_id}{self.sep}{image_id}.png'

    def _get_bbox(self, patient_id, image_id):
        if self.bbox is not None:
            key = f'{patient_id}/{image_id}.png'
            if key in self.bbox.keys():
                bbox = self.bbox[key]
                bbox = [bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax'], 'YOLO']
            else:
                bbox = [0, 0, 100000, 100000, 'YOLO'] # dummy
        else:
            bbox = None
        return bbox

    def _load_best_image(self, df): # for test
        scores = []
        images = []
        bboxes = []
        if 'implant' in df.columns:
            is_implant = df['implant'].values[0]
        else:
            is_implant = 0
        for pid, iid in df[['patient_id', 'image_id']].values:
            img_path = self._get_file_path(pid, iid)
            bbox = self._get_bbox(pid, iid)
            img = self._load_image(img_path)
            bboxes.append(bbox)
            scores.append(img.mean())
            images.append(img)
        if is_implant and self.sample_criteria == 'low_value_for_implant':
            score_idx = np.argmin(scores)
        else:
            score_idx = np.argmax(scores)
        return images[score_idx], bboxes[score_idx]

    def _load_data(self, idx):
        pid = self.pids[idx]
        pdf_lr = self.df_dict[pid]
        imgs = {}
        labels = {}
        for laterality in ['L', 'R']:
            pdf = pdf_lr.loc[pdf_lr['laterality'] == laterality]
            img_v = []
            for iv, view_cat in enumerate(self.view_category):
                view0 = pdf.loc[pdf['view'].isin(view_cat)]
                if len(view0) == 0:
                    img0 = np.zeros((self.img_size, self.img_size, 1), dtype=np.uint8)
                    bbox0 = self._get_bbox('none', 'none')
                    img0 = self._preprocess_img(img0, bbox0)
                    img_v.append(img0)
                else:
                    if self.is_test:
                        img0, bbox0 = self._load_best_image(view0)
                        img0 = self._preprocess_img(img0, bbox0)
                        img_v.append(img0)
                    else:
                        view0 = view0.sample(1)
                        for pid, iid in view0[['patient_id', 'image_id']].values:
                            img_path = self._get_file_path(pid, iid)
                            bbox0 = self._get_bbox(pid, iid)
                            img0 = self._load_image(img_path)
                            img0 = self._preprocess_img(img0, bbox0)
                            img_v.append(img0)
            img = np.stack(img_v, axis=0)
            imgs[laterality] = img
            labels[laterality] = torch.from_numpy(pdf[self.target_cols+self.aux_target_cols].values[0].astype(np.float16))
        
        if self.transform_imagewise:
            img_size = imgs['L'].shape[1:]
            imgs = np.stack([imgs['L'], imgs['R']], axis=1).squeeze(-1).reshape(-1, *img_size) # (L, R, L, R, ...)
            imgs = torch.cat([self._augment_img(img) for img in imgs], dim=0)
        else:
            imgs = np.stack([imgs['L'], imgs['R']], axis=3).squeeze(-1) # (view, h, w, laterality) 
            imgs = torch.cat([self._augment_img(img) for img in imgs], dim=0)

        _, h, w = imgs.shape
        imgs = imgs.view(-1, 2, h, w) # (view, lat, h, w)
        if self.separate_channel:
            imgs = imgs.unsqueeze(2) # (view, lat, 1, h, w)
        labels = torch.stack([labels['L'], labels['R']], dim=0)

        return imgs, labels

    def __getitem__(self, idx):
        imgs, labels = self._load_data(idx)

        if self.rt_idx:
            return imgs, labels, idx
        else:
            return imgs, labels

    def get_labels(self):
        labels = []
        for idx in range(len(self.df_dict)):
            pid = self.pids[idx]
            pdf = self.df_dict[pid]
            for laterality in ['L', 'R']:
                labels.append(
                    pdf.loc[pdf['laterality'] == laterality, self.target_cols].values[0].reshape(1, 1).astype(np.float16))
        return np.concatenate(labels, axis=0)
