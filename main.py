import logging
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import tyro
from model.faster_rcnn.resnet import resnet
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import (  # (1) here add a function to viz
    vis_detections_filtered_objects,
    vis_detections_filtered_objects_PIL,
)
from torch import Tensor
from tqdm.rich import tqdm_rich as tqdm

from utils import AVC1MP4Writer, read_media, set_logging_level, write_pickle

log = logging.getLogger()

pascal_classes = np.asarray(["__background__", "targetobject", "hand"])


@dataclass
class Args:
    dataset: str = "pascal_voc"
    """training dataset"""
    cfg_file: str = "cfgs/res101.yml"
    """optional config file"""
    net: str = "res101"
    """you can only choose res101"""
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
    thresh_hand: float = 0.5
    """hand threshold"""
    thresh_obj: float = 0.5
    """object threshold"""
    log_level: str = "info"
    """log level"""
    sample_intv: int = 1
    """sample interval"""
    max_out_frames: int = math.inf
    """maximum number of frames to process"""
    overwrite: bool = False
    """overwrite existing results"""


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


def load_model(args: Args, cfg) -> nn.Module:
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

    # initilize the network here.
    if args.net == "res101":
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False)
    else:
        raise ValueError(f"network {args.net} is not defined")

    fasterRCNN.create_architecture()

    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint["model"])
    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    log.info(f"load checkpoint {load_name} successfully!")
    return fasterRCNN


def preprocess_image(img: np.ndarray):
    blobs, im_scales = _get_image_blob(img)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32
    )

    im_data = torch.from_numpy(im_blob).permute(0, 3, 1, 2).to(device)
    im_info = torch.from_numpy(im_info_np).to(device)
    return im_data, im_info, im_scales


def detect(
    args: Args,
    cfg,
    fasterRCNN: nn.Module,
    im_data: Tensor,
    im_info: Tensor,
    im_scales: np.ndarray,
):
    ## Dummy values
    num_boxes = torch.zeros(1)
    gt_boxes = torch.zeros((1, 1, 5)).to(device)
    box_info = torch.zeros((1, 1, 5)).to(device)

    ## Forward pass
    tic = time.time()
    with torch.inference_mode():
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
    toc = time.time()
    forward_time = toc - tic
    log.debug(f"Forward pass took {forward_time:.3f}s")

    ## Postprocessing predicted results
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
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                cfg.TRAIN.BBOX_NORMALIZE_STDS
            ).to(device) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(device)
            box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    pred_boxes = pred_boxes.squeeze()  # (300, 12)
    scores = scores.squeeze()  # (300, 3)
    return pred_boxes, scores, contact_indices, offset_vector, lr


def do_NMS(
    args: Args,
    cfg,
    pred_boxes: Tensor,
    scores: Tensor,
    contact_indices: Tensor,
    offset_vector: Tensor,
    lr: Tensor,
):
    obj_dets, hand_dets = None, None
    for j in range(1, len(pascal_classes)):
        pascal_class = pascal_classes[j]
        if pascal_class == "hand":
            inds = torch.nonzero(scores[:, j] > args.thresh_hand).view(-1)
        elif pascal_class == "targetobject":
            inds = torch.nonzero(scores[:, j] > args.thresh_obj).view(-1)

        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)

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
            # * TIPS: comment below line to see all detections
            cls_dets = cls_dets[keep.view(-1).long()]
            if pascal_class == "targetobject":
                obj_dets = cls_dets.cpu().numpy()
            if pascal_class == "hand":
                hand_dets = cls_dets.cpu().numpy()

    return obj_dets, hand_dets


def main(args: Args, cfg):
    np.random.seed(cfg.RNG_SEED)

    fasterRCNN = load_model(args, cfg)
    fasterRCNN.to(device)
    fasterRCNN.eval()

    log.info(f"Reading videos from {args.image_dir}")
    media_list = [
        v
        for v in os.listdir(args.image_dir)
        if v.endswith(".mp4") or v.endswith(".png")
    ]
    num_videos = len(media_list)
    log.info(f"Loaded {num_videos} images and videos")

    for video_idx, video_name in (pbar := tqdm(list(enumerate(media_list)))):
        src_video_path = os.path.join(args.image_dir, video_name)
        save_video_path = os.path.join(args.save_dir, f"{video_name[:-4]}_det.mp4")
        if os.path.exists(save_video_path) and not args.overwrite:
            log.info(f"Skipping {video_name}")
            continue
        os.makedirs(args.save_dir, exist_ok=True)
        writer = AVC1MP4Writer(
            save_video_path,
            fps=30 / args.sample_intv,
        )

        frames = read_media(
            src_video_path,
            sample_intv=args.sample_intv,
            max_out_frames=args.max_out_frames,
        )
        log.info(
            f"Read {len(frames)} frames from {video_name} with sample interval {args.sample_intv}"
        )

        # Process frames
        output_results = []
        for frame_idx, frame in enumerate(frames):
            # if frame_idx >= 5:
            #     break
            # Preprocess
            im_data, im_info, im_scales = preprocess_image(frame)

            # Detect
            det_tic = time.time()
            pred_boxes, scores, contact_indices, offset_vector, lr = detect(
                args, cfg, fasterRCNN, im_data, im_info, im_scales
            )
            det_toc = time.time()
            detect_time = det_toc - det_tic

            # NMS and visualize
            nms_tic = time.time()
            obj_dets, hand_dets = do_NMS(
                args, cfg, pred_boxes, scores, contact_indices, offset_vector, lr
            )
            # img_show_cv2 = vis_detections_filtered_objects(
            #     frame, obj_dets, hand_dets, thresh=0.5
            # )
            img_show = vis_detections_filtered_objects_PIL(
                frame, obj_dets, hand_dets, args.thresh_hand, args.thresh_obj
            )
            img_show = np.array(img_show)
            img_show = img_show[..., :3]
            nms_toc = time.time()
            nms_time = nms_toc - nms_tic

            # Append to output
            writer.write(img_show)
            output_results.append({"obj_dets": obj_dets, "hand_dets": hand_dets})

            # Profiling
            total_time = detect_time + nms_time
            pbar.set_description(
                " ".join(
                    [
                        f"Processed media {video_idx + 1}/{num_videos}, frame {frame_idx + 1}/{len(frames)} in {total_time:.2f}s:",
                        f"Detect={detect_time:.2f}s",
                        f"NMS={nms_time:.2f}s",
                    ]
                )
            )

        # Save results
        writer.release()
        write_pickle(
            output_results, os.path.join(args.save_dir, f"{video_name[:-4]}.pkl")
        )


if __name__ == "__main__":
    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    args = tyro.cli(Args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.cuda > 0:
        cfg.CUDA = True
    cfg.USE_GPU_NMS = args.cuda

    device = torch.device("cuda" if args.cuda else "cpu")
    set_logging_level(args.log_level)
    main(args, cfg)
