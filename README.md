# EMBC-Net
Official implementation of EMBC-Net, an edge-aware multi-scale collaborative enhancement network with bidirectional semantic calibration for colorectal polyp segmentation.
# Architecture
<img width="1260" height="709" alt="fig1" src="https://github.com/user-attachments/assets/1bdc122b-fc0e-4c01-8d6d-6bdcc30819e0" />
# Usage:
Recommended environment:
Python 3.8
Pytorch 1.11.0
torchvision 0.12.0
# Data preparation:
Polyp datasets: Download training and testing datasets from [Google Drive]([[https://drive.google.com/xxxxx](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?pli=1)])) and move them into './data/polyp/'.
# Training:
For Polyp training run CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py
# Testing:
For Polyp testing run CUDA_VISIBLE_DEVICES=0 python -W ignore test_polyp.py
