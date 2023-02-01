import torch.nn as nn
import torch.nn.functional as F
import timm
from kuma_utils.torch.modules import AdaptiveConcatPool2d, AdaptiveGeM

from modules import *


class ClassificationModel(nn.Module):
    def __init__(self,
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=1,
                 num_classes=1,
                 custom_classifier='none',
                 custom_attention='none', 
                 dropout=0,
                 pretrained=False):

        super().__init__()

        self.encoder = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0, '')

        if custom_attention == 'triplet':
            self.attention = TripletAttention()
        else:
            self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = AdaptiveGeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if 'Transformer' in self.encoder.__class__.__name__:
            self.encoder.patch_embed = CustomHybdridEmbed(
                self.encoder.patch_embed.proj, 
                channel_in=in_chans,
                transformer_original_input_size=(1, in_chans, *self.encoder.patch_embed.img_size),
                pretrained=pretrained
            )
            self.is_tranformer = True
        else:
            self.is_tranformer = False

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(feature_dim, 512), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, num_classes))

    def forward(self, x):
        output = self.encoder(x)
        output = self.attention(output)
        if self.is_tranformer:
            output = output.mean(dim=1)
        else:
            output = self.global_pool(output)
        output = self.head(output)
        return output


class MultiViewModel(nn.Module):
    def __init__(self,
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=1,
                 num_classes=1,
                 num_view=2,
                 custom_classifier='none',
                 custom_attention='none', 
                 dropout=0,
                 hidden_dim=1024,
                 spatial_pool=False,
                 pretrained=False):

        super().__init__()

        self.encoder = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0, '')

        if custom_attention == 'triplet':
            self.attention = TripletAttention()
        else:
            self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = AdaptiveGeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if 'Transformer' in self.encoder.__class__.__name__:
            self.encoder.patch_embed = CustomHybdridEmbed(
                self.encoder.patch_embed.proj, 
                channel_in=in_chans,
                transformer_original_input_size=(1, in_chans, *self.encoder.patch_embed.img_size),
                pretrained=pretrained
            )
            self.is_tranformer = True
        else:
            self.is_tranformer = False

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim*num_view, hidden_dim) if not spatial_pool else nn.Linear(feature_dim, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim//2), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim//2, num_classes))
        
        self.spatial_pool = spatial_pool

    def forward(self, x): # (N x n_views x Ch x W x H)
        bs, n_view, ch, w, h = x.shape
        x = x.view(bs*n_view, ch, w, h)
        y = self.attention(self.encoder(x)) # (2n_views x Ch2 x W2 x H2)
        if self.is_tranformer:
            y = y.mean(dim=1).view(bs, n_view, -1).mean(dim=1)
        else:
            if self.spatial_pool:
                _, ch2, w2, h2 = y.shape
                y = y.view(bs, n_view, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
                    .contiguous().view(bs, ch2, n_view*w2, h2)
            y = self.global_pool(y) # (bs x Ch2 x 1 x 1)
        if not self.spatial_pool:
            y = y.view(bs, n_view, -1)
        y = self.head(y)
        return y

    
class MultiViewSiameseModel(nn.Module):

    def __init__(self,
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=1,
                 num_classes=1,
                 num_view=2,
                 custom_classifier='none',
                 custom_attention='none', 
                 dropout=0,
                 hidden_dim=1024,
                 pretrained=False):

        super().__init__()

        self.encoder = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0, '')

        if custom_attention == 'triplet':
            self.attention = TripletAttention()
        else:
            self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = AdaptiveGeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if 'Transformer' in self.encoder.__class__.__name__:
            self.encoder.patch_embed = CustomHybdridEmbed(
                self.encoder.patch_embed.proj, 
                channel_in=in_chans,
                transformer_original_input_size=(1, in_chans, *self.encoder.patch_embed.img_size),
                pretrained=pretrained
            )
            self.is_tranformer = True
        else:
            self.is_tranformer = False

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim*num_view*2, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim//2), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim//2, num_classes))

    def forward(self, x):
        bs, n_view, ch, w, h = x.shape
        x = x.view(bs*n_view, ch, w, h)
        y = self.attention(self.encoder(x)) # (2n_views x Ch2 x W2 x H2)
        y = self.global_pool(y) # (bs x Ch2 x 1 x 1)
        y = y.view(bs, n_view, -1)
        y_s = torch.cat([torch.abs(y[:, 0, :] - y[:, 1, :]), y[:, 0, :] * y[:, 1, :]], dim=1)
        y = torch.cat([y.view(bs, -1), y_s], dim=1)
        y = self.head(y)
        return y


class MultiViewSiameseLRModel(nn.Module):

    def __init__(self,
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=1,
                 num_classes=1,
                 num_view=2,
                 custom_classifier='none',
                 custom_attention='none', 
                 dropout=0,
                 hidden_dim=1024,
                 pretrained=False):

        super().__init__()

        self.encoder = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0, '')

        if custom_attention == 'triplet':
            self.attention = TripletAttention()
        else:
            self.attention = nn.Identity()

        if custom_classifier == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif custom_classifier == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif custom_classifier == 'concat':
            self.global_pool = AdaptiveConcatPool2d()
            feature_dim = feature_dim * 2
        elif custom_classifier == 'gem':
            self.global_pool = AdaptiveGeM(p=3, eps=1e-4)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if 'Transformer' in self.encoder.__class__.__name__:
            self.encoder.patch_embed = CustomHybdridEmbed(
                self.encoder.patch_embed.proj, 
                channel_in=in_chans,
                transformer_original_input_size=(1, in_chans, *self.encoder.patch_embed.img_size),
                pretrained=pretrained
            )
            self.is_tranformer = True
        else:
            self.is_tranformer = False

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(feature_dim*num_view*3, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim//2), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim//2, num_classes))

    def forward(self, x): # (bs, view, lat, h, w)
        bs, n_view, n_lat, ch, w, h = x.shape
        x = x.view(bs*n_view*n_lat, ch, w, h)
        y = self.attention(self.encoder(x)) # (2 n_view n_lat x C2 x W2 x H2)
        y = self.global_pool(y).view(bs*n_view, n_lat, -1) # (bs n_view x n_lat x C2)
        y_l = y[:, 0, :]
        y_r = y[:, 1, :]
        y_s = torch.cat([torch.abs(y_l - y_r), y_l * y_r], dim=1) # (bs n_view x 2 C2)
        y_sl = torch.cat([y_s, y_l], dim=1) # (bs n_view x 3 C2)
        y_sr = torch.cat([y_s, y_r], dim=1) 
        y_sl = y_sl.view(bs, n_view, -1) # (bs x 3 n_view C2)
        y_sr = y_sr.view(bs, n_view, -1)
        y_l = self.head(y_sl)
        y_r = self.head(y_sr)
        y = torch.cat([y_l, y_r], dim=1).view(bs*n_lat, 1)
        return y


class MultiInstanceModel(MultiViewModel):

    def __init__(self, **kwargs):
        kwargs['spatial_pool'] = True
        super().__init__(**kwargs)

    def forward(self, x): # N x 2 x n_inst x C x W x H
        bs, n_view, n_inst, ch, w, h = x.shape
        x = x.view(bs*n_view*n_inst, ch, w, h)
        y = self.attention(self.encoder(x)) # (2Nn_inst x Ch2 x 1 x 1)
        if self.is_tranformer: # TODO
            _, ch2, f = y.shape
            y = y.view(bs, n_view*n_inst*ch2, f).mean(dim=1)
        else:
            _, ch2, w2, h2 = y.shape
            y = y.view(bs, n_view*n_inst, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
                .contiguous().view(bs, ch2, n_view*n_inst*w2, h2)
            y = self.global_pool(y) 
        y = self.head(y)
        return y
    

class MultiLevelModel(nn.Module):

    def __init__(self,
                 global_model='resnet18',
                 global_model_params={},
                 local_model='resnet18',
                 local_model_params={},
                 in_chans=1,
                 num_classes=1,
                 num_view=2,
                 hidden_dim=512,
                 dropout=0, 
                 crop_size=256,
                 crop_num=4,
                 percent_t=0.02,
                 local_attention=False,
                 pretrained=False):

        super().__init__()

        self.global_encoder = timm.create_model(
            global_model,
            pretrained=pretrained,
            in_chans=in_chans,
            **global_model_params
        )
        global_feature_dim = self.global_encoder.get_classifier().in_features
        self.global_encoder.reset_classifier(0, '')
        if 'Transformer' in self.global_encoder.__class__.__name__:
            self.is_transformer_global = True
        else:
            self.is_transformer_global = False
        self.local_encoder = timm.create_model(
            local_model,
            pretrained=pretrained,
            in_chans=in_chans,
            **local_model_params
        )
        local_feature_dim = self.local_encoder.get_classifier().in_features
        self.local_encoder.reset_classifier(0, '')
        if 'Transformer' in self.local_encoder.__class__.__name__:
            self.is_transformer_local = True
        else:
            self.is_transformer_local = False
        if local_attention:
            self.local_attention = TripletAttention()
        else:
            self.local_attention = nn.Identity()

        self.local_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(local_feature_dim*num_view, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes))
        self.concat_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear((global_feature_dim+local_feature_dim)*num_view, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes))
        

        self.localizer = nn.Conv2d(global_feature_dim, num_classes, (1, 1), bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.crop_size = crop_size
        self.crop_num = crop_num
        self.percent_t = percent_t
        self.num_view = num_view

    def crop_roi(self, x, cam): # current implementation is for 1 class only
        crop_size = self.crop_size
        crop_num = self.crop_num
        bs, ch, h0, w0 = x.shape
        # _, ch1, h1, w1 = cam.shape
        cam_full = F.interpolate(cam, size=(h0, w0), mode='nearest')
        x2 = torch.concat([x, cam_full], dim=1)
        pad_h, pad_w = (crop_size - h0 % crop_size) % crop_size, (crop_size - w0 % crop_size) % crop_size
        x2 = F.pad(
            x2, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2, 0, 0), 
            mode='constant', value=0)
        _, _, h2, w2 = x2.shape
        x2 = x2.view(bs, ch+1, h2//crop_size, crop_size, w2//crop_size, crop_size)
        x2 = x2.permute(0, 2, 4, 1, 3, 5).contiguous()
        x2 = x2.view(bs, -1, ch+1, crop_size, crop_size)
        score = torch.mean(x2[:, :, -1:,  :, :], dim=(2, 3, 4))
        x2 = x2[:, :, :-1, :, :] # drop cam
        score_sort = torch.argsort(score, dim=1, descending=True)
        x2 = x2[torch.arange(bs)[:, None], score_sort]
        x2 = x2[:, :crop_num].clone()
        return x2

    def aggregate_cam(self, cam):
        bs, ch, h, w = cam.shape
        cam_flatten = cam.view(bs, ch, -1)
        top_t = int(round(w*h*self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]
        return selected_area.mean(dim=2)

    def forward_mil(self, x):
        bs, n_inst, ch, w, h = x.shape
        x = x.view(bs*n_inst, ch, w, h)
        y = self.local_encoder(x) # (2Nn_inst x Ch2 x W x H) or
        if self.is_transformer_local:
            _, ch2, f = y.shape
            y = y.view(bs, n_inst, ch2, f).view(bs, ch2*n_inst, f).mean(dim=1)
        else:
            _, ch2, w2, h2 = y.shape
            y = y.view(bs, n_inst, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
                .contiguous().view(bs, ch2, n_inst*w2, h2)
            y = self.global_pool(self.local_attention(y))
        return y

    def squeeze_view(self, x):
        bs, _, ch, w, h = x.shape
        x = x.view(bs*self.num_view, ch, w, h)
        return x

    def forward(self, x): # (N x n_views x Ch x W x H)
        bs = x.shape[0]
        if self.num_view > 1:
            x = self.squeeze_view(x)
        global_features = self.global_encoder(x) # (Nn_views x Ch2 x W2 x H2)
        global_cam= self.localizer(global_features) # (Nn_views x 1 x W2 x H2)
        x2 = self.crop_roi(x, global_cam.sigmoid()) # (Nn_views x n_crop x 1 x Wl x Hl)
        local_features = self.forward_mil(x2)
        global_features = self.global_pool(global_features)
        y_global = self.aggregate_cam(global_cam)
        if self.num_view > 1:
            local_features = local_features.view(bs, self.num_view, -1) # (N x n_views x Ch3)
            global_features = global_features.view(bs, self.num_view, -1) # (N x n_views x Ch2)
            y_global = y_global.view(bs, self.num_view, -1).amax(dim=1) # 
        concat_features = torch.concat([local_features, global_features], dim=2) # (bs x n_views x Ch2+Ch3)
        y_local = self.local_head(local_features)
        y_concat = self.concat_head(concat_features)
        return y_concat, y_global, y_local


class SegAssistModel(nn.Module):
    def __init__(self,
                 classification_model='resnet18',
                 classification_params={},
                 in_chans=1,
                 map_channel=3,
                 num_classes=1,
                 dropout=0,
                 pretrained=False,
                 seg_only=False):

        super().__init__()

        self.encoder = timm.create_model(
            classification_model,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **classification_params
        )
        feature_dim = self.encoder.get_classifier().in_features
        self.encoder.reset_classifier(0, '')
        self.localizer = nn.Conv2d(feature_dim, map_channel, (1, 1), bias=False)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(feature_dim, 512), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, num_classes))
        self.seg_only = seg_only

    def forward(self, x):
        output = self.encoder(x)
        seg_map = self.localizer(output)
        if self.seg_only:
            return seg_map.sigmoid()
        else:
            output = self.global_pool(output)
            output = self.head(output)
            return output, seg_map.sigmoid()
        