from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import tyro
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import (  # (1) here add a function to viz
    load_net,
    save_net,
    vis_detections,
    vis_detections_filtered_objects,
    vis_detections_filtered_objects_PIL,
    vis_detections_PIL,
)
from PIL import Image
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from torch.autograd import Variable

pascal_classes = np.asarray(["__background__", "targetobject", "hand"])


@dataclass
class Args:
    """Train a Fast R-CNN network"""

    dataset: str = "pascal_voc"
    """training dataset"""
    cfg_file: str = "cfgs/res101.yml"
    """optional config file"""
    net: str = "res101"
    """vgg16, res50, res101, res152"""
    set_cfgs: Optional[List[str]] = None
    """set config keys"""
    load_dir: str = "models"
    """directory to load models"""
    image_dir: str = "images"
    """directory to load images for demo"""
    save_dir: str = "images_det"
    """directory to save results"""
    cuda: bool = True
    """whether use CUDA"""
    mGPUs: bool = False
    """whether use multiple GPUs"""
    class_agnostic: bool = False
    """whether perform class_agnostic bbox regression"""
    parallel_type: int = 0
    """which part of model to parallel, 0: all, 1: model before roi pooling"""
    checksession: int = 1
    """checksession to load model"""
    checkepoch: int = 8
    """checkepoch to load network"""
    checkpoint: int = 132028
    """checkpoint to load network"""
    batch_size: int = 1
    """batch_size"""
    vis: bool = True
    """visualization mode"""
    thresh_hand: float = 0.5
    """hand threshold"""
    thresh_obj: float = 0.5
    """object threshold"""


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(
            im_orig,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def load_model(args, cfg):
    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception(
            "There is no input directory for loading network from " + model_dir
        )
    load_name = os.path.join(
        model_dir,
        "faster_rcnn_{}_{}_{}.pth".format(
            args.checksession, args.checkepoch, args.checkpoint
        ),
    )

    args.set_cfgs = ["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"]

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(
            pascal_classes, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res101":
        fasterRCNN = resnet(
            pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res50":
        fasterRCNN = resnet(
            pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic
        )
    elif args.net == "res152":
        fasterRCNN = resnet(
            pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint["model"])
    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    print("load model successfully!")
    return fasterRCNN


if __name__ == "__main__":
    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    args = tyro.cli(Args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    fasterRCNN = load_model(args, cfg)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand
        thresh_obj = args.thresh_obj
        vis = args.vis

        print(f"image dir = {args.image_dir}")
        print(f"save dir = {args.save_dir}")
        imglist = os.listdir(args.image_dir)
        imglist = [
            img for img in imglist if img.endswith(".png") or img.endswith(".jpg")
        ]
        num_images = len(imglist)

        print("Loaded Photo: {} images.".format(num_images))

        while num_images >= 0:
            total_tic = time.time()
            num_images -= 1

            # Load the demo image
            im_file = os.path.join(args.image_dir, imglist[num_images])
            im_in = cv2.imread(im_file)
            im = im_in

            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
            )

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_()

            # pdb.set_trace()
            det_tic = time.time()

            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls,
                rpn_loss_box,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                loss_list,
            ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extact predicted params
            contact_vector = loss_list[0][0]  # hand contact state info
            offset_vector = loss_list[1][
                0
            ].detach()  # offset vector (factored into a unit vector and a magnitude)
            lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

            # get hand contact
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        if args.cuda > 0:
                            box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS
                                ).cuda()
                                + torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_MEANS
                                ).cuda()
                            )
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS
                            ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if args.cuda > 0:
                            box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS
                                ).cuda()
                                + torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_MEANS
                                ).cuda()
                            )
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS
                            ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im2show = np.copy(im)
            obj_dets, hand_dets = None, None
            for j in range(1, len(pascal_classes)):
                # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                if pascal_classes[j] == "hand":
                    inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
                elif pascal_classes[j] == "targetobject":
                    inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

                    cls_dets = torch.cat(
                        (
                            cls_boxes,
                            cls_scores.unsqueeze(1),
                            contact_indices[inds],
                            offset_vector.squeeze(0)[inds],
                            lr[inds],
                        ),
                        1,
                    )
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if pascal_classes[j] == "targetobject":
                        obj_dets = cls_dets.cpu().numpy()
                    if pascal_classes[j] == "hand":
                        hand_dets = cls_dets.cpu().numpy()

            if vis:
                # visualization
                im2show = vis_detections_filtered_objects_PIL(
                    im2show, obj_dets, hand_dets, thresh_hand, thresh_obj
                )

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write(
                "im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r".format(
                    num_images + 1, len(imglist), detect_time, nms_time
                )
            )
            sys.stdout.flush()

            if vis:

                folder_name = args.save_dir
                os.makedirs(folder_name, exist_ok=True)
                result_path = os.path.join(
                    folder_name, imglist[num_images][:-4] + "_det.png"
                )
                im2show.save(result_path)
