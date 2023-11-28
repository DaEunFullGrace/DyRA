import argparse, math, os, copy
from typing import List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Boxes
from detectron2.layers import cat

from .match_predictor import SimplePredictor
from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..meta_arch import META_ARCH_REGISTRY

__all__ = ["DynamicResizer"]


def weight_func(logits, tau=1):
    weight = tau / torch.clamp(tau - logits.detach(), min=1e-2)
    weight /= weight.sum(dim=-1).unsqueeze(dim=-1)
    return weight

def get_least_one(x):
    x *= x.numel()
    return x - x.min() + 1


@META_ARCH_REGISTRY.register()
class DynamicResizer(nn.Module): 
    @configurable
    def __init__(self, *, 
                 net: SimplePredictor, 
                 image_encoder, 
                 class_num, 
                 feat_sizes,
                 pixel_mean, 
                 pixel_std, 
                 out_layer=["res4"]):
        super().__init__()
        
        self.predictor = net
        self.image_encoder = image_encoder
        self.class_num = class_num
        self.out_feat = out_layer
        
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        
        self.grid_sizes = torch.tensor(feat_sizes).to(self.device)
        self.coco_bnd = torch.tensor([32.**2, 96.**2], requires_grad=False).to(self.device)
        self.mapping_ratio = nn.Parameter((self.grid_sizes[2] ** 2) / self.coco_bnd.mean())
        
    @property
    def device(self):
        return self.predictor.device
    
    @classmethod
    def from_config(cls, cfg):
        image_encoder = build_backbone(cfg, is_resizer=True)
        
        ## for class_num
        return {
            "net":SimplePredictor(in_chan=15,
                                  device=cfg.MODEL.DEVICE,
                                  num_mlayer=cfg.MODEL.RESIZER.ENCODER.RES2_OUT_CHANNELS*4),
            "image_encoder": image_encoder,
            "out_layer": cfg.MODEL.RESIZER.ENCODER.OUT_FEATURES,
            "class_num":cfg.MODEL.RETINANET.NUM_CLASSES, 
            "feat_sizes": [32., 64., 128., 256., 512.], 
            "pixel_mean": cfg.MODEL.PIXEL_MEAN, 
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], get_sf=False, debug=False):
        sc_facs, size_dub = [], []
        
        for i, x in enumerate(batched_inputs):
            image = self.preprocess_image(x, self.pixel_mean, self.pixel_std)
            img_feat = self.image_encoder(image.tensor)[self.out_feat[0]]
            sc_sf = self.predictor(img_feat)
            sc_facs.append(sc_sf.squeeze())
            
            if "instances" in x:
                gt_boxes = x["instances"].get_fields()["gt_boxes"].tensor.to(self.device)
                box_sizes = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                size_dub.append((-self.map_range(box_sizes,
                                                 self.coco_bnd*self.mapping_ratio)).sigmoid().mean())
            
        if self.training:
            loss_dict = self.loss(batched_inputs, torch.stack(sc_facs))
            
            if debug:
                get_event_storage().put_scalar("sf", sc_facs[0].item())
                get_event_storage().put_scalar("s_bnd", self.mapping_ratio.item())
                get_event_storage().put_scalar("l_bnd", self.mapping_ratio.item())
                if "instances" in batched_inputs[0]:
                    get_event_storage().put_scalar("size_ratio", size_dub[0].item())
            
            if get_sf:    return dict(loss_dict, **{"sf": sc_facs})
            else:    return loss_dict
        else:    return {"sf": sc_facs}
        
    def loss(self, batched_inputs, sc_sf):
        assert not torch.any(torch.isnan(sc_sf))
        max_ious, sc_losses = [], []
        
        for i, x in enumerate(batched_inputs):
            gt_boxes = x["instances"].get_fields()["gt_boxes"].tensor.detach().to(self.device)
            scaled_boxes = gt_boxes * sc_sf[i].detach()
            # box_sizes = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            scaled_sizes = (scaled_boxes[:, 2] - scaled_boxes[:, 0]) * (scaled_boxes[:, 3] - scaled_boxes[:, 1])
            del scaled_boxes, gt_boxes
            
            size_ratios = (-self.map_range(scaled_sizes, self.coco_bnd*self.mapping_ratio)).sigmoid()
            sc_bnd = self.predictor.gain
            sc_loss = - (size_ratios*((sc_sf[i]/sc_bnd).clamp(min=1e-12).log()) + 
                        (1-size_ratios)*((1-sc_sf[i]/sc_bnd).clamp(min=1e-12)).log())
            
            # box size pareto
            pareto_sc = [sc_loss[torch.where((scaled_sizes>=bnd[0]) & (scaled_sizes<bnd[1]))].mean() 
                         for bnd in zip(self.grid_sizes, self.grid_sizes[1:])]
            pareto_sc.append(sc_loss[torch.where(scaled_sizes<self.grid_sizes[0])].mean())
            pareto_sc.append(sc_loss[torch.where(scaled_sizes>=self.grid_sizes[-1])].mean())
            pareto_sc = torch.stack(pareto_sc, dim=0)
            pareto_sc = -(((-pareto_sc[torch.where(pareto_sc.isnan()==False)]).exp()).prod()).log()
            sc_losses.append(pareto_sc)

        return {"loss_ps": torch.stack(sc_losses, dim=0).mean()}
    
    def balance_loss(self, anchors: List[Union[Boxes, torch.Tensor]],
                     box2box_transform: Box2BoxTransform,
                     boxes: List[torch.Tensor],
                     fg_mask: torch.Tensor, loc_loss: torch.Tensor, reduction='mean'):
        
        if anchors is not None:
            if isinstance(anchors[0], Boxes):
                anchors = type(anchors[0]).cat(anchors).tensor
            else:
                anchors = cat(anchors)
            
            box_deltas = [box2box_transform.get_deltas(anchors, k) for k in boxes]
            box_deltas = torch.stack(box_deltas)[fg_mask]
        else:
            box_deltas = boxes
        
        if box_deltas.size()[-1] == 4:
            box_sizes = (box_deltas[:, 3] - box_deltas[:, 1]) * (box_deltas[:, 2] - box_deltas[:, 0])
        else:
            box_sizes = boxes.detach()
        del box_deltas
        
        ac_size_mean = (self.coco_bnd * self.mapping_ratio).mean()
        if reduction == "sum":
            losses_group = torch.stack([loc_loss[torch.where(box_sizes<ac_size_mean)].sum(), 
                                        loc_loss[torch.where(box_sizes>=ac_size_mean)].sum()])
        else:
            losses_group = torch.stack([loc_loss[torch.where(box_sizes<ac_size_mean)].mean(), 
                                        loc_loss[torch.where(box_sizes>=ac_size_mean)].mean()])
            
        loc_weight = weight_func(losses_group.nan_to_num(nan=(losses_group[losses_group.isnan()==False][0].item())))
        loc_weight = get_least_one(loc_weight)
        
        mean_bnd = self.grid_sizes[1::2] ** 2
        cur_mean_ratio = (self.mapping_ratio * self.coco_bnd).mean() / self.coco_bnd.mean()
        tar_mean_ratio = (loc_weight.flip(-1) * mean_bnd).mean() / self.coco_bnd.mean()
    
        return {"loss_bal": F.l1_loss(cur_mean_ratio, tar_mean_ratio)}
    
    def preprocess_image(self, x: Dict[str, torch.Tensor], pixel_mean, pixel_std):
        """
        Normalize, pad and batch the input images.
        """
        image = (x["image"]).to(self.device)
        image = (image - pixel_mean) / pixel_std
        image = ImageList.from_tensors([image,])
        return image
        
    def map_range(self, x, xrange, target_range=[-6, 6]):
        div_num = (xrange[1] - xrange[0]) / (target_range[1] - target_range[0])
        rng_min = target_range[0] - xrange[0] / div_num
        return x / div_num + rng_min
