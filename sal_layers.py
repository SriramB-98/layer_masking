import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv2d


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
            return self._conv_forward(input, self.weight, self.bias)
        elif isinstance(input, tuple):
            inp, mask, it = input
            self.filt = self.filt.to(mask.device)
            if it[0] >= 0:
                pad = self.weight.shape[-1] if it[0]>0 else 0
                masked_inp = fast_fmm(inp, mask, self.filt, iters=pad)
#                 it[0] = it[0]-1
            else:
                masked_inp = inp
            
            conv_inp = self._conv_forward(masked_inp, self.weight, self.bias)
            
            self.ones = self.ones.to(mask.device)
            
            if it[0] < 0:
                c = self._conv_forward(mask, self.ones, None) + 1e-8
                conv_inp = conv_inp*(1/c) 
                if self.bias is not None:
                    conv_inp += self.bias*(1 - 1/c)
            
            with torch.no_grad():
                conv_mask = self.maxpool(mask)
            return (conv_inp*conv_mask, conv_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
        
class MyMaxPool2d(nn.MaxPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            pool_inp = super().forward(inp)#*mask
            with torch.no_grad():
                pool_mask = super().forward(mask)
            return (pool_inp*pool_mask, pool_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyAvgPool2d(nn.AvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            pool_inp = super().forward(inp)#*mask
            with torch.no_grad():
                pool_mask = (super().forward(mask) > 0).float()
            return (pool_inp*pool_mask, pool_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
    
class MyBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            out = super().forward(inp*mask)
            mask_avg = super().forward(mask) + 1e-8
            return out/mask_avg
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MySigmoid(nn.Sigmoid):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyReLU(nn.ReLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

            
class MyHardsigmoid(nn.Hardsigmoid):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyHardswish(nn.Hardswish):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MySiLU(nn.SiLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyGELU(nn.GELU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp), mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv2d


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
            return self._conv_forward(input, self.weight, self.bias)
        elif isinstance(input, tuple):
            inp, mask, it = input
            self.filt = self.filt.to(mask.device)
            if it[0] >= 0:
                pad = self.weight.shape[-1] if it[0]>0 else 0
                masked_inp = fast_fmm(inp, mask, self.filt, iters=pad)
            else:
                masked_inp = inp
            
            conv_inp = self._conv_forward(masked_inp, self.weight, self.bias)
            
            self.ones = self.ones.to(mask.device)
            
            if it[0] < 0:
                c = self._conv_forward(mask, self.ones, None) + 1e-8
                conv_inp = conv_inp*(1/c) 
                if self.bias is not None:
                    conv_inp += self.bias*(1 - 1/c)
            
            with torch.no_grad():
                conv_mask = self.maxpool(mask)
            return (conv_inp*conv_mask, conv_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
        
class MyMaxPool2d(nn.MaxPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            pool_inp = super().forward(inp)#*mask
            with torch.no_grad():
                pool_mask = super().forward(mask)
            return (pool_inp*pool_mask, pool_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyAvgPool2d(nn.AvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            pool_inp = super().forward(inp)#*mask
            with torch.no_grad():
                pool_mask = (super().forward(mask) > 0).float()
            return (pool_inp*pool_mask, pool_mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
    
class MyBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            out = super().forward(inp*mask)
            mask_avg = super().forward(mask) + 1e-8
            return out/mask_avg
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MySigmoid(nn.Sigmoid):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyReLU(nn.ReLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')

            
class MyHardsigmoid(nn.Hardsigmoid):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyHardswish(nn.Hardswish):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MySiLU(nn.SiLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp*mask)*mask, mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyGELU(nn.GELU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp), mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyIdentity(nn.Identity):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, it = input
            return (super().forward(inp), mask, it)
        else:
            raise Exception(f'Encountered input type {type(input)}')