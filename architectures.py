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
                transformer_original_input_size=(1, in_chans, *self.encoder.patch_embed.img_size)
            )
            self.is_tranformer = True
        else:
            self.is_tranformer = False

        self.head = nn.Sequential(
            Flatten(),
            nn.Linear(feature_dim*2, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim//2), 
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim//2, num_classes))

    def forward(self, x0, x1):
        y0 = self.encoder(x0)
        y0 = self.attention(y0)
        if self.is_tranformer:
            y0 = y0.mean(dim=1)
        else:
            y0 = self.global_pool(y0)
        
        y1 = self.encoder(x1)
        y1 = self.attention(y1)
        if self.is_tranformer:
            y1 = y1.mean(dim=1)
        else:
            y1 = self.global_pool(y1)

        output = torch.concat([y0, y1], dim=1)
        output = self.head(output)
        return output
