import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError


def cluster_iou(annotation, segmentation):
    annotation = annotation.astype(np.int64)
    assert segmentation.size == annotation.size
    return np.sum((annotation & segmentation)) / np.sum((annotation | segmentation), dtype=np.float32)


def mask_assignment(annotation, segmentation, n_clusters):
    mask = np.zeros(segmentation.shape)
    for i in range(n_clusters):
        s = np.where(segmentation == i, True, False)
        if cluster_iou(annotation, s) >= cluster_iou(~annotation.astype(np.bool), s):
            mask[s] = 1
    mask = mask.astype(np.int64)

    return mask


def mask_save(mask, save_dir, save_size):
    mask = cv2.resize(mask, (save_size[1], save_size[0]), interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float)
    mask *= 255.0
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(save_dir)


def get_uv_grid(h, w, align_corners=False, device=None):
    if device is None:
        device = torch.device('cpu')
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
    )

    if align_corners:
        xx = 2 * xx / (w - 1) - 1
        yy = 2 * yy / (h - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / w - 1
        yy = 2 * (yy + 0.5) / h - 1

    return torch.stack([xx, yy], dim=-1)


def get_flow_coords(flo, align_corners=False):
    device = flo.device
    *dims, _, h, w = flo.shape
    uv = get_uv_grid(h, w, align_corners=align_corners, device=device)
    uv = uv.view(*(1,) * len(dims), h, w, 2)
    flo = flo.permute(0, 2, 3, 1)
    if align_corners:
        flo[..., 0] = 2.0 * flo[..., 0].clone() / max(w - 1, 1)
        flo[..., 1] = 2.0 * flo[..., 1].clone() / max(h - 1, 1)
    else:
        flo[..., 0] = 2.0 * flo[..., 0].clone() / max(w, 1)
        flo[..., 1] = 2.0 * flo[..., 1].clone() / max(h, 1)
    return uv + flo


def inverse_flow_warp(x, flo, align_corners=False):
    vgrid = get_flow_coords(flo, align_corners=align_corners)
    output = F.grid_sample(x, vgrid, mode='nearest', align_corners=align_corners, padding_mode='border')
    return output
