import torch.nn as nn
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
                transformer_original_input_size=(1, in_chans, *self.encoder.patch_embed.img_size)
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
                transformer_original_input_size=(1, in_chans, *self.encoder.patch_embed.img_size)
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
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_view = num_view
        self.spatial_pool = spatial_pool

    def forward(self, x): # (N x n_views x Ch x W x H)
        bs, n_view, ch, w, h = x.shape
        x = x.view(bs*n_view, ch, w, h)
        y = self.attention(self.encoder(x)) # (2n_views x Ch2 x W2 x H2)
        if self.is_tranformer: # TODO
            y = y.mean(dim=1)
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


class MultiInstanceModel(MultiViewModel):

    def __init__(self, **kwargs):
        kwargs['spatial_pool'] = True
        super().__init__(**kwargs)

    def forward(self, x): # N x 2 x n_inst x C x W x H
        bs, n_view, n_inst, ch, w, h = x.shape
        x = x.view(bs*n_view*n_inst, ch, w, h)
        y = self.attention(self.encoder(x)) # (2Nn_inst x Ch2 x 1 x 1)
        if self.is_tranformer: # TODO
            y = y.mean(dim=1)
        else:
            _, ch2, w2, h2 = y.shape
            y = y.view(bs, n_view*n_inst, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
                .contiguous().view(bs, ch2, n_view*n_inst*w2, h2)
            y = self.global_pool(y) 
        y = self.head(y)
        return y
    