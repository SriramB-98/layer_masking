from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.init as init
from torch.nn.functional import conv2d

try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401

from torchvision.utils import _log_api_usage_once

__all__ = ["SqueezeNet", "squeezenet1_0", "squeezenet1_1"]

model_urls = {
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
}

def fast_fmm(img, mask, filt, iters=4, eps=1e-5):
    img = img*mask
    c = img.shape[1]
    filtc = filt.expand(c, -1, -1, -1)
    for i in range(iters):
        with torch.no_grad():
            num = conv2d(img, filtc, groups=c, padding='same')
            denom = conv2d(mask, filt, groups=1, padding='same')
            edge_fills = (1-mask)*(num/(denom+eps))
        img = img + edge_fills.detach()
        mask = (mask + denom).clip(0,1)
    return img

class MyReLU(nn.ReLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp), mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
        
        
class MyConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h, w = self.kernel_size
        self.ones = torch.ones(1, 1, h, w, device=self.weight.device)/(h*w)
        self.filt = torch.ones(1, 1, 3, 3, device=self.weight.device)
        self.maxpool = torch.nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=self.padding)
        return
    
    def forward(self, input):
        if isinstance(input, Tensor):
            out = self._conv_forward(input, self.weight, self.bias)
            return out
        elif isinstance(input, tuple):
            inp, mask, it = input
            self.filt = self.filt.to(mask.device)
            with torch.no_grad():
                conv_mask = self.maxpool(mask)
            masked_inp = fast_fmm(inp, mask, self.filt, iters=self.weight.shape[-1])
            conv_inp = self._conv_forward(masked_inp, self.weight, self.bias)#[None,:,None,None]
            return (conv_inp, conv_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')      

class MyMaxPool2d(nn.MaxPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            pool_inp = super().forward(inp*mask)#
            with torch.no_grad():
                pool_mask = super().forward(mask)
            return (pool_inp, pool_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = MyConv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = MyReLU(inplace=True)
        self.expand1x1 = MyConv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = MyReLU(inplace=True)
        self.expand3x3 = MyConv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = MyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        if isinstance(x, Tensor):
            return torch.cat([self.expand1x1_activation(self.expand1x1(x)),self.expand3x3_activation(self.expand3x3(x))], 1)
        elif isinstance(x, tuple):
            im_a, mask_a, it = self.expand1x1_activation(self.expand1x1(x))
            im_b, mask_b, it = self.expand3x3_activation(self.expand3x3(x))
            mask = torch.clip(mask_a + mask_b, 0, 1)
            return (torch.cat([im_a,im_b], 1), mask, it)

class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return super().forward(inp*mask)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyDropout(nn.Dropout):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp), mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class SqueezeNet(nn.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                MyConv2d(3, 96, kernel_size=7, stride=2),
                MyReLU(inplace=True),
                MyMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                MyMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                MyMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                MyConv2d(3, 64, kernel_size=3, stride=2),
                MyReLU(inplace=True),
                MyMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                MyMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                MyMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = MyConv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            MyDropout(p=dropout), final_conv, MyReLU(inplace=True), MyAdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = "squeezenet" + version
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    The required minimum input size of the model is 21x21.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet("1_0", pretrained, progress, **kwargs)


def squeezenet1_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    The required minimum input size of the model is 17x17.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet("1_1", pretrained, progress, **kwargs)
