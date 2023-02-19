import torch
import torch.nn as nn
from timm import create_model
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin


class ConvNextSmallEncoder(nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels = [3, 96, 96, 192, 384, 768] # output channels
        self._depth = 5 # UNet depth
        self._in_channels = 1
        self._m = create_model(
            model_name='convnext_small.fb_in22k_ft_in1k_384',
            in_chans=self._in_channels,
            pretrained=False,
        )
        # self._m.reset_classifier(0, '')

    def get_stages(self):
        return [
            nn.Identity(),
            self._m.stem,
            self._m.stages[0],
            self._m.stages[1],
            self._m.stages[2],
            self._m.stages[3],
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict):
        state_dict['model'].pop('head.weight')
        state_dict['model'].pop('head.bias')
        self._m.load_state_dict(state_dict['model'], strict=False)


class ConvNextTinyEncoder(nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels = [3, 96, 96, 192, 384, 768] # output channels
        self._depth = 5 # UNet depth
        self._in_channels = 1
        self._m = create_model(
            model_name='convnext_tiny.fb_in22k_ft_in1k_384',
            in_chans=self._in_channels,
            pretrained=False,
        )
        # self._m.reset_classifier(0, '')

    def get_stages(self):
        return [
            nn.Identity(),
            self._m.stem,
            self._m.stages[0],
            self._m.stages[1],
            self._m.stages[2],
            self._m.stages[3],
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict):
        state_dict['model'].pop('head.weight')
        state_dict['model'].pop('head.bias')
        self._m.load_state_dict(state_dict['model'], strict=False)


class ResNet200dEncoder(nn.Module, EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels = [3, 64, 256, 512, 1024, 2048] # output channels
        self._depth = 5 # UNet depth
        self._in_channels = 3
        self._m = create_model(
            model_name='resnet200d',
            in_chans=self._in_channels,
            pretrained=False
        )
        self._m.global_pool = nn.Identity()
        self._m.fc = nn.Identity()

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._m.conv1, self._m.bn1, self._m.act1),
            nn.Sequential(self._m.maxpool, self._m.layer1),
            self._m.layer2,
            self._m.layer3,
            self._m.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        self._m.load_state_dict(state_dict, strict=False)


smp.encoders.encoders["convnext_small"] = {
    "encoder": ConvNextSmallEncoder,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth", 
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {}
}


smp.encoders.encoders["convnext_tiny"] = {
    "encoder": ConvNextTinyEncoder,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth", 
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {}
}


smp.encoders.encoders["resnet200d"] = {
    "encoder": ResNet200dEncoder,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth", # this can be the pretrained weight from timm
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {}
}