# DyRA

This repository is an implemented version of DyRA based on detectron2.\
Our work supports RetinaNet, Faster-RCNN, Mask-RCNN, FCOS, and DETR(still in training).

| Model | AP | AP_l | AP_m | AP_s | weights |
|---|---|---|---|---|---|
RetinaNet-ResNet50 | 40.1 | 52.5 | 43.4 | 24.8 | | 
RetinaNet-ResNet101 | 41.5 | 53.7 | 45.2 | 25.3 | | 
FasterRCNN-ResNet50 | 41.2 | 54.4 | 44.1 | 25.0 | | 
FasterRCNN-ResNet101 | 43.1 | 57.1 | 46.1 | 26.9 | |
MaskRCNN-ResNet50 | 41.8 | 54.7 | 44.5 | 26.0 | |
MaskRCNN-ResNet101 | 43.6 | 57.8 | 46.8 | 26.3 | | 
FCOS-ResNet50 | 42.3 | 54.2 | 45.5 | 26.4 | | 
FCOS-ResNet101 | 43.8 | 56.4 | 47.4 | 28.8 | |

## Installation
```
git clone https://github.com/NotCaffee/DyRA.git
python -m pip install -e detectron2
python -m pip install -e AdelaiDet
```
</code></pre>
DETR - Will be uploaded

## Config Files
Config files are in "detectron2/configs/DyRA".\
If you want to use pre-trained weight for DyRA's image encoder, you should check the cfg.RESIZER.WEIGHTS contains a "resizer".\
END_LR of ConstCosineLR: 3/2 * cfg.BASE_LR_END

## Training and Evaluation
The same command as the detectron2.
