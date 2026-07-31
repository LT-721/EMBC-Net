# EMBC-Net

Official implementation of EMBC-Net, an edge-aware multi-scale collaborative enhancement network with bidirectional semantic calibration for colorectal polyp segmentation.

# Architecture

<img width="1260" height="709" alt="fig1" src="https://github.com/user-attachments/assets/1bdc122b-fc0e-4c01-8d6d-6bdcc30819e0" />

# Usage

Recommended environment: Python 3.8, PyTorch 1.11.0, torchvision 0.12.0.<br>
Polyp datasets can be downloaded from [Google Drive](https://drive.google.com/file/d/1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC/view?pli=1) and should be placed in the `./data/polyp/` directory.<br>
The PVTv2-B2 backbone is initialized with ImageNet-1K pre-trained weights.<br>


Training command: `CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py`.<br>


Testing command: `CUDA_VISIBLE_DEVICES=0 python -W ignore test_polyp.py`.

# License

This project is released under the MIT License.
