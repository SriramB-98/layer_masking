from itertools import chain
import torch
import torchvision
from torchvision import datasets, transforms
import os
import random
import numpy as np
import pandas as pd
import json
import time

from PIL import Image
import pickle
from collections import OrderedDict, Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import hsv_to_rgb

import PIL

import linecache

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    
# import sys
# sys.path.insert(-1, '../')

# from data_utils import CustomDataSet


def update_time(t):
    print(f"time elapsed: {time.time() - t}")
    return time.time()

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dict_to_str(dic):
    s = ''
    for k, it in dic.items():
        s += str(k)+'-'+str(it)+'_'
    return s[:-1]

def list_to_str(lis, sep='\n'):
    lis = [str(x) for x in lis]
    return sep.join(lis)

def ce(bv):
    return np.sum(-bv*np.log(bv + 1e-7), axis=-1)

def expand_and_flatten(model_list):
    if not model_list:
        return []
    flattened_list = []
    for model in model_list:
        child_list = expand_and_flatten(list(model.children()))
        if not child_list and list(model.parameters()):
            child_list = [model]
        flattened_list.extend(child_list)
    return flattened_list

def to_numpy(tens):
    return tens.cpu().detach().numpy()

def colorgrid(array, norm=0.1):
    red = np.zeros(shape=array.shape)
    blue = 240/360*np.ones(shape=array.shape)
    hue = red*(array > 0) + blue*(array<=0)
    sat = np.clip(np.abs(array)/norm, -1, 1)
    val = np.ones(shape=array.shape)
    hsv_img = np.stack((hue, sat, val), axis=-1)
    return hsv_to_rgb(hsv_img), hsv_img
    
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def freeze(model_part):
    for param in model_part.parameters():
        param.requires_grad = False

def unfreeze(model_part):
    for param in model_part.parameters():
        param.requires_grad = True