# DyRA: Dynamic Resolution Adjustment for Scale-robust Object Detection
* [PaperUpdate] Currently updating [PaperLink](https://arxiv.org/abs/2311.17098) - To be more easy to read, almost done.
* [CodeImplementation] There is a code implementation issue for balance loss. Currently, we are fixing it (+ Re-experimenting for this).

Our work supports RetinaNet, Faster-RCNN, Mask-RCNN, FCOS, and DETR(still in training).

| Model | AP | AP_l | AP_m | AP_s | Weights | Log |
|---|---|---|---|---|---|---|
RetinaNet-ResNet50 | 40.1 | 52.5 | 43.4 | 24.8 | | |
RetinaNet-ResNet101 | 41.5 | 53.7 | 45.2 | 25.3 | | |
FasterRCNN-ResNet50 | 41.2 | 54.4 | 44.1 | 25.0 | | |
FasterRCNN-ResNet101 | 43.1 | 57.1 | 46.1 | 26.9 | | |
MaskRCNN-ResNet50 | 41.8 | 54.7 | 44.5 | 26.0 | | |
MaskRCNN-ResNet101 | 43.6 | 57.8 | 46.8 | 26.3 | | |
FCOS-ResNet50 | 42.3 | 54.2 | 45.5 | 26.4 | | |
FCOS-ResNet101 | 43.8 | 56.4 | 47.4 | 28.8 | | |

For anchor-based detectors, test augmentations are maximum=minimum=800.

## Installation
```
git clone https://github.com/DaEunFullGrace/DyRA.git
python -m pip install -e detectron2
python -m pip install -e AdelaiDet
```
DETR - Will be uploaded together with the weight link

## Config Files
* Config files are in "detectron2/configs/DyRA" or "AdelaiDet/configs/DyRA"
* To use pre-trained weight for DyRA's image encoder, you should check the cfg.RESIZER.WEIGHTS contains a "resizer"
  * E.g., cfg.RESIZER.WEIGHTS = "../weights/R-50-resizer.pkl"
* END_LR of ConstCosineLR: approximately 3/2 * cfg.BASE_LR_END
  * cfg.BASE_LR_END=7e-5 - can get higher accuracy in certain models(like RetinaNet or FCOS)

## Training and Evaluation
The same command as the detectron2

## Acknowledgement
Our code is based on [detectron2](https://github.com/facebookresearch/detectron2) and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet/tree/master).
