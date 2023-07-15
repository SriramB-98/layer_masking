First install the prerequisites in requirements.txt

Then, if you want to run layer masking on timm resnet (the data augmented one robust to greyout), replace timm/models/resnet.py with timm_resnet.py and rename it to resnet.py. Also, place a copy of sal_layers.py in timm/models . 

Then run the jupyter notebooks:
1. Analysis: Segment removal experiments
2. Mask shape bias: Analysis of shape bias of masking tecnique
3. LIME (quantitative/qualitative): Experiments on LIME
4. Measure Linearity: Linearity experiments

Make sure to insert the correct paths for Salient ImageNet and Pixel ImageNet in the notebooks and code