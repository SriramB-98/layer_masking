from typing import Any

import torch
from torch import nn, Tensor
from torch.nn.functional import conv2d


try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url 
    
from torchvision.utils import _log_api_usage_once

__all__ = ["AlexNet", "alexnet"]


model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
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
            conv_inp = self._conv_forward(masked_inp, self.weight, self.bias)
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
    
class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return super().forward(inp*mask)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyReLU(nn.ReLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp), mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
        
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = nn.Sequential(
            MyConv2d(3, 64, kernel_size=11, stride=4, padding=2),
            MyReLU(inplace=True),
            MyMaxPool2d(kernel_size=3, stride=2),
            MyConv2d(64, 192, kernel_size=5, padding=2),
            MyReLU(inplace=True),
            MyMaxPool2d(kernel_size=3, stride=2),
            MyConv2d(192, 384, kernel_size=3, padding=1),
            MyReLU(inplace=True),
            MyConv2d(384, 256, kernel_size=3, padding=1),
            MyReLU(inplace=True),
            MyConv2d(256, 256, kernel_size=3, padding=1),
            MyReLU(inplace=True),
            MyMaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = MyAdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
        model.load_state_dict(state_dict)
    return model
