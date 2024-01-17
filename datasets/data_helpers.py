import os

import cv2
import einops
import numpy as np
from PIL import Image


def make_color_wheel(bins=None):
    """Build a color wheel

    Args:
        bins(list or tuple, optional): specify number of bins for each color
            range, corresponding to six ranges: red -> yellow, yellow -> green,
            green -> cyan, cyan -> blue, blue -> magenta, magenta -> red.
            [15, 6, 4, 11, 13, 6] is used for default (see Middlebury).

    Returns:
        ndarray: color wheel of shape (total_bins, 3)
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T


def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image

    Args:
        flow(ndarray): optical flow
        color_wheel(ndarray or None): color wheel used to map flow field to RGB
            colorspace. Default color wheel will be used if not specified
        unknown_thr(str): values above this threshold will be marked as unknown
            and thus ignored

    Returns:
        ndarray: an RGB image that can be visualized
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) | (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    [h, w] = dx.shape

    rad = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (
        1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img


TAG_FLOAT = 202021.25


def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def readFlow(sample_dir, resolution, to_rgb):
    flow = read_flo(sample_dir)
    h, w, _ = np.shape(flow)
    flow = cv2.resize(flow, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] = flow[:, :, 0] * resolution[1] / w
    flow[:, :, 1] = flow[:, :, 1] * resolution[0] / h
    if to_rgb:
        flow = np.clip((flow2rgb(flow) - 0.5) * 2, -1., 1.)
    return einops.rearrange(flow, 'h w c -> c h w')


def readRGB(sample_dir, resolution, normalize):
    rgb = cv2.imread(sample_dir, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # rgb = ((rgb / 255.0) - 0.5) * 2.0
    rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    if normalize:
        rgb = np.clip((rgb / 255.0 - 0.5) * 2, -1., 1.)
    # rgb = np.clip(rgb, -1., 1.)
    return einops.rearrange(rgb, 'h w c -> c h w')


def readRGBUINT8(sample_dir, resolution=None, transpose=True):
    rgb = cv2.imread(sample_dir, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if resolution is not None:
        if rgb.shape[:-1] != resolution:
            rgb = cv2.resize(rgb, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    if transpose:
        return einops.rearrange(rgb, 'h w c -> c h w')
    return rgb


def readSeg(sample_dir, resolution=None):
    # gt = cv2.imread(sample_dir)
    seg = Image.open(sample_dir).convert('P')
    seg = np.array(seg, dtype=np.int32)
    # return einops.rearrange(gt, 'h w c -> c h w')
    if resolution is not None:
        if seg.shape != resolution:
            seg = cv2.resize(seg, (resolution[1], resolution[0]), interpolation=cv2.INTER_NEAREST)
    return seg


def np2img(arr):
    if len(arr.shape) == 2:
        mode = 'P'
    else:
        mode = 'RGB'

    return Image.fromarray(arr, mode=mode)
