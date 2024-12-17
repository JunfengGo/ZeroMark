# ZeroMark
This is the official implementation of our paper [ZeroMark: Towards Dataset Ownership Verification without Disclosing Watermark](), accepted by NeurIPS 2024. This research project is developed based on Python 3 and Pytorch, created by [Junfeng Guo](https://junfenggo.github.io/).

# Dependencies

Our code is implemented using Torch. Following packages are required.

```bash
PyTorch => 1.6.*
torchvision > 0.5.*
Our code is tested on Python 3.8.3
```

# Implementatiion

To generate the boundary gradients for samples belong to (target) class 0 uisng samples from (ori) class 1, run: 

```bash
python zeromark.py --ori 1 --target 0
```

to get the normalized boundary gradient similarity by:

```bash
python ga.py
```


