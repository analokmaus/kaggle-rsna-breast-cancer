import torch
import torch.nn as nn
from timm.layers.conv_bn_act import ConvBnAct
import timm
import torch
from torch import nn


def mixup(img, lam, idx): # img: (N, C, H, W)
    return img * lam[:, None, None, None] + img.index_select(0, idx) * (1-lam)[:, None, None, None], lam


def mixup2d(img, lam, idx):
    return img * lam[:, None, None] + img.index_select(0, idx) * (1-lam)[:, None, None], lam


def mixup1d(img, lam, idx):
    return img * lam[:, None] + img.index_select(0, idx) * (1-lam)[:, None], lam


def get_bbox(shape, lam):
    bs, _, w, h = shape
    ratio = torch.sqrt(1. - lam)
    cut_w = (w * ratio).int()
    cut_h = (h * ratio).int()

    cx = torch.randint(0, w, (bs, ), device=lam.device)
    cy = torch.randint(0, h, (bs, ), device=lam.device)

    x1 = (cx - cut_w // 2).clamp(0, w)
    y1 = (cy - cut_h // 2).clamp(0, h)
    x2 = (cx + cut_w // 2).clamp(0, w)
    y2 = (cy + cut_h // 2).clamp(0, h)
    return x1, y1, x2, y2


def cutmix(img, lam, idx): # img: (N, C, H, W)
    x1, y1, x2, y2 = get_bbox(img.shape, lam)
    img2 = img.index_select(0, idx)
    for i in range(img.shape[0]):
        img[i, :, x1[i]:x2[i], y1[i]:y2[i]] = img2[i, :, x1[i]:x2[i], y1[i]:y2[i]]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (img.shape[2] * img.shape[3])
    return img, lam


class ManifoldMixup(nn.Module):

    def __init__(self):
        super().__init__()
        self.lam = None
        self.idx = None

    def forward(self, x):
        if x.requires_grad:
            return mixup(x, self.lam, self.idx)[0]
        else:
            return x

    def update(self, lam, idx):
        self.lam = lam
        self.idx = idx


class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x): 
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)


class BidirectionalLSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden_dim, bidirectional=True)
        self.embed = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, input):
        output, _ = self.rnn(input)
        dim, bs, h = output.shape
        output = output.view(dim * bs, h)
        output = self.embed(output) 
        output = output.view(dim, bs, -1)
        return output


class Upsample(nn.Module):
    def __init__(self, in_channels, target_size, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        self.target_size = target_size
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=self.target_size, mode="bilinear")
        if self.with_conv:
            x = self.conv(x)
        return x


class CustomHybdridEmbed(nn.Module):
    def __init__(
        self,
        decoder_proj_conv,
        channel_in=2,
        encoder_name="tf_efficientnet_b0",
        # encoder_name="inception_v4",
        encoder_out_layer_num=[2],
        transformer_original_input_size=(1, 3, 224, 224),
        pretrained=False,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            out_indices=encoder_out_layer_num,
            pretrained=pretrained,
            in_chans=channel_in,
        )

        with torch.no_grad():
            x = torch.rand(1, channel_in, 360, 512)
            enc_ch_num = self.encoder(x)[0].shape[1]
            decoder_in_channels = decoder_proj_conv(
                torch.rand(transformer_original_input_size)
            ).shape

        self.resize = Upsample(enc_ch_num, decoder_in_channels[2:])
        self.proj = nn.Conv2d(
            enc_ch_num, decoder_in_channels[1], kernel_size=1, stride=1
        )

    def forward(self, x):
        x = self.encoder(x)[0]
        x = self.resize(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


""" Triplet Attention Module
Implementation of triplet attention module from https://arxiv.org/abs/2010.03045
(slightly) Modified from official implementation: https://github.com/LandskapeAI/triplet-attention
Original license:
MIT License
Copyright (c) 2020 LandskapeAI
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(AttentionGate, self).__init__()
        self.zpool = ZPool()
        self.conv = ConvBnAct(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2, apply_act=False)
    
    def forward(self, x):
        x_out = self.conv(self.zpool(x))
        scale = torch.sigmoid_(x_out) 
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = nn.Identity() if no_spatial else AttentionGate()
        self.no_spatial = no_spatial

    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        x_out = self.hw(x)

        x_out = (1/2) * (x_out11 + x_out21) if self.no_spatial else (1/3) * (x_out + x_out11 + x_out21)
        return x_out
