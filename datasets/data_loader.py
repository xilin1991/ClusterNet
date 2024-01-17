import os

import einops
import cv2
# from PIL import Image
# import numpy as np
import torch
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
from torch.utils import data

import datasets.data_helpers as helpers


class SequenceLoader(data.Dataset):
    def __init__(self, root_dir, seq_name, resolution, to_rgb=False, with_gt=False):
        super(SequenceLoader, self).__init__()

        self.root_dir = root_dir
        self.seq_name = seq_name
        self.resolution = resolution
        self.to_rgb = to_rgb
        self.with_gt = with_gt

        flo_list = []
        seg_list = []

        flo_flies = sorted(os.listdir(os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq_name)))
        flo_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq_name, x), flo_flies))
        flo_list.extend(flo_paths)

        seg_flies = sorted(os.listdir(os.path.join(root_dir, 'Annotations_unsupervised/480p', seq_name)))[:-1]
        seg_paths = list(map(lambda x: os.path.join(root_dir, 'Annotations_unsupervised/480p', seq_name, x), seg_flies))
        seg_list.extend(seg_paths)

        assert len(flo_list) == len(seg_list)
        self.flo_list = flo_list
        self.seg_list = seg_list
        print('Done Initializing {0} of DAVIS Dataset.'.format(seq_name))

    def __len__(self):
        return len(self.flo_list)

    def __getitem__(self, index):
        flo_file = self.flo_list[index]
        seg_file = self.seg_list[index]

        img_path = os.path.splitext(seg_file)[0]
        img_path, img_name = os.path.split(img_path)
        seq_name = os.path.split(img_path)[-1]

        flow = helpers.readFlow(flo_file, self.resolution, self.to_rgb)
        seg = helpers.readSeg(seg_file)
        ori_size = seg.shape
        if self.resolution is not None:
            if seg.shape != self.resolution:
                seg = cv2.resize(seg, (self.resolution[1], self.resolution[0]),
                                 interpolation=cv2.INTER_NEAREST)

        flow = torch.from_numpy(flow).float()
        seg = torch.from_numpy(seg).long()

        seg[seg == 255] = 0
        seg[seg != 0] = 1
        seg = seg.float()
        seg = einops.rearrange(seg, 'h w -> (h w)')

        meta = {'seq_name': seq_name, 'img_name': img_name, 'ori_size': ori_size}
        sample = {'flow': flow, 'meta': meta}
        if self.with_gt:
            sample['seg'] = seg

        return sample


class SequenceLoader_FBMS(data.Dataset):
    def __init__(self, root_dir, seq_name, resolution, to_rgb=False, with_gt=False):
        super(SequenceLoader_FBMS, self).__init__()

        seq_name = seq_name.strip()

        self.root_dir = root_dir
        self.seq_name = seq_name
        self.resolution = resolution
        self.to_rgb = to_rgb
        self.with_gt = with_gt

        flo_list = []
        seg_list = []

        seg_flies = sorted(os.listdir(os.path.join(root_dir, 'Annotations_Binary', seq_name)))[:-1]
        seg_paths = list(map(lambda x: os.path.join(root_dir, 'Annotations_Binary', seq_name, x), seg_flies))
        seg_list.extend(seg_paths)

        flo_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_1_FlowFormer', seq_name, x[:-7] + '.flo'), seg_flies))
        flo_list.extend(flo_paths)

        assert len(flo_list) == len(seg_list)
        self.flo_list = flo_list
        self.seg_list = seg_list
        print('Done Initializing {0} of DAVIS Dataset.'.format(seq_name))

    def __len__(self):
        return len(self.flo_list)

    def __getitem__(self, index):
        flo_file = self.flo_list[index]
        seg_file = self.seg_list[index]

        img_path = os.path.splitext(seg_file)[0]
        img_path, img_name = os.path.split(img_path)
        seq_name = os.path.split(img_path)[-1]

        seg = helpers.readSeg(seg_file)
        ori_size = seg.shape
        flow = helpers.readFlow(flo_file, self.resolution, self.to_rgb)
        # if self.resolution is not None:
        #     if seg.shape != self.resolution:
        #         seg = cv2.resize(seg, (self.resolution[1], self.resolution[0]),
        #                          interpolation=cv2.INTER_NEAREST)

        flow = torch.from_numpy(flow).float()
        seg = torch.from_numpy(seg).long()

        seg[seg != 0] = 1
        seg = seg.float()
        seg = einops.rearrange(seg, 'h w -> (h w)')

        meta = {'seq_name': seq_name, 'img_name': img_name, 'ori_size': ori_size}
        sample = {'flow': flow, 'meta': meta}
        if self.with_gt:
            sample['seg'] = seg

        return sample


class SequenceLoader_SegTrack(data.Dataset):
    def __init__(self, root_dir, seq_name, resolution, to_rgb=False, with_gt=False):
        super(SequenceLoader_SegTrack, self).__init__()

        seq_name = seq_name.strip()

        self.root_dir = root_dir
        self.seq_name = seq_name
        self.resolution = resolution
        self.to_rgb = to_rgb
        self.with_gt = with_gt

        flo_list = []
        seg_list = []

        flo_flies = sorted(os.listdir(os.path.join(root_dir, 'Flows_gap_1_FlowFormer', seq_name)))
        flo_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_1_FlowFormer', seq_name, x), flo_flies))
        flo_list.extend(flo_paths)

        seg_flies = sorted(os.listdir(os.path.join(root_dir, 'Annotations_Binary', seq_name)))[:-1]
        seg_paths = list(map(lambda x: os.path.join(root_dir, 'Annotations_Binary', seq_name, x), seg_flies))
        seg_list.extend(seg_paths)

        assert len(flo_list) == len(seg_list)
        self.flo_list = flo_list
        self.seg_list = seg_list
        print('Done Initializing {0} of DAVIS Dataset.'.format(seq_name))

    def __len__(self):
        return len(self.flo_list)

    def __getitem__(self, index):
        flo_file = self.flo_list[index]
        seg_file = self.seg_list[index]

        img_path = os.path.splitext(seg_file)[0]
        img_path, img_name = os.path.split(img_path)
        seq_name = self.seq_name

        seg = helpers.readSeg(seg_file)
        ori_size = seg.shape
        flow = helpers.readFlow(flo_file, self.resolution, self.to_rgb)
        # if self.resolution is not None:
        #     if seg.shape != self.resolution:
        #         seg = cv2.resize(seg, (self.resolution[1], self.resolution[0]),
        #                          interpolation=cv2.INTER_NEAREST)

        flow = torch.from_numpy(flow).float()
        seg = torch.from_numpy(seg).long()

        seg[seg != 0] = 1
        seg = seg.float()
        seg = einops.rearrange(seg, 'h w -> (h w)')

        meta = {'seq_name': seq_name, 'img_name': img_name, 'ori_size': ori_size}
        sample = {'flow': flow, 'meta': meta}
        if self.with_gt:
            sample['seg'] = seg

        return sample


class SequenceLoader_Consistency(data.Dataset):
    def __init__(self, root_dir, seq_name, resolution, to_rgb=False, with_gt=False):
        super(SequenceLoader_Consistency, self).__init__()

        seq_name = seq_name.strip()

        self.root_dir = root_dir
        self.seq_name = seq_name
        self.resolution = resolution
        self.to_rgb = to_rgb
        self.with_gt = with_gt

        flo1_list = []
        flo2_list = []
        seg_list = []

        flo1_flies = sorted(os.listdir(os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq_name)))
        flo1_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq_name, x), flo1_flies))
        flo1_list.extend(flo1_paths)

        flo2_flies = sorted(os.listdir(os.path.join(root_dir, 'Flows_gap_r1_FlowFormer/Full-Resolution', seq_name)))
        flo2_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_r1_FlowFormer/Full-Resolution', seq_name, x), flo2_flies))
        flo2_list.extend(flo2_paths)

        seg_flies = sorted(os.listdir(os.path.join(root_dir, 'Annotations_unsupervised/480p', seq_name)))[:-1]
        seg_paths = list(map(lambda x: os.path.join(root_dir, 'Annotations_unsupervised/480p', seq_name, x), seg_flies))
        seg_list.extend(seg_paths)

        assert len(flo1_list) == len(flo2_list) == len(seg_list)
        self.flo1_list = flo1_list
        self.flo2_list = flo2_list
        self.seg_list = seg_list
        print('Done Initializing {0} of DAVIS Dataset.'.format(seq_name))

    def __len__(self):
        return len(self.flo1_list)

    def __getitem__(self, index):
        flo1_file = self.flo1_list[index]
        flo2_file = self.flo2_list[index]
        seg_file = self.seg_list[index]

        img_path = os.path.splitext(seg_file)[0]
        img_path, img_name = os.path.split(img_path)
        seq_name = os.path.split(img_path)[-1]

        flow1 = helpers.readFlow(flo1_file, self.resolution, self.to_rgb)
        flo1 = helpers.readFlow(flo1_file, self.resolution, to_rgb=False)
        flow2 = helpers.readFlow(flo2_file, self.resolution, self.to_rgb)
        flo2 = helpers.readFlow(flo2_file, self.resolution, to_rgb=False)
        seg = helpers.readSeg(seg_file)
        ori_size = seg.shape
        if self.resolution is not None:
            if seg.shape != self.resolution:
                seg = cv2.resize(seg, (self.resolution[1], self.resolution[0]),
                                 interpolation=cv2.INTER_NEAREST)

        flow1 = torch.from_numpy(flow1).float()
        flow2 = torch.from_numpy(flow2).float()
        # flow = torch.stack([flow1, flow2], dim=0)
        seg = torch.from_numpy(seg).long()

        seg[seg == 255] = 0
        seg[seg != 0] = 1
        seg = seg.float()
        seg = einops.rearrange(seg, 'h w -> (h w)')

        meta = {'seq_name': seq_name, 'img_name': img_name, 'ori_size': ori_size}
        sample = {'flow1': flow1, 'flow2': flow2, 'flo1': flo1, 'flo2': flo2, 'meta': meta}
        if self.with_gt:
            sample['seg'] = seg

        return sample


class FlowLoader(data.Dataset):
    def __init__(self, root_dir, resolution, split='train', year=2016,
                 to_rgb=False, with_gt=False):
        super(FlowLoader, self).__init__()

        self.root_dir = root_dir
        self.resolution = resolution
        self.split = split
        self.year = year
        self.to_rgb = to_rgb
        self.with_gt = with_gt

        if isinstance(year, int):
            year = str(year)

        with open(os.path.join(root_dir, 'ImageSets', year, '{}.txt'.format(split))) as f:
            seqs = f.readlines()

        flo_list = []
        seg_list = []

        for seq in seqs:
            seq = seq.strip()

            flo_files = sorted(os.listdir(os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq)))
            flo_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq, x), flo_files))
            flo_list.extend(flo_paths)

            seg_files = sorted(os.listdir(os.path.join(root_dir, 'Annotations_unsupervised/480p', seq)))[:-1]
            seg_paths = list(map(lambda x: os.path.join(root_dir, 'Annotations_unsupervised/480p', seq, x), seg_files))
            seg_list.extend(seg_paths)

        assert len(flo_list) == len(seg_list)
        self.flo_list = flo_list
        self.seg_list = seg_list
        print('Done Initializing {0} set of DAVIS {1} Dataset.'.format(split, year))

    def __len__(self):
        return len(self.flo_list)

    def __getitem__(self, index):
        flo_file = self.flo_list[index]
        seg_file = self.seg_list[index]

        img_path = os.path.splitext(seg_file)[0]
        img_path, img_name = os.path.split(img_path)
        seq_name = os.path.split(img_path)[-1]

        flow = helpers.readFlow(flo_file, self.resolution, self.to_rgb)

        seg = helpers.readSeg(seg_file)
        ori_size = seg.shape
        if self.resolution is not None:
            if seg.shape != self.resolution:
                seg = cv2.resize(seg, (self.resolution[1], self.resolution[0]),
                                 interpolation=cv2.INTER_NEAREST)

        flow = torch.from_numpy(flow).float()
        seg = torch.from_numpy(seg).long()

        seg[seg == 255] = 0
        if self.year == '2016':
            seg[seg != 0] = 1
        seg = seg.float()
        seg = einops.rearrange(seg, 'h w -> (h w)')

        meta = {'seq_name': seq_name, 'img_name': img_name, 'ori_size': ori_size}
        sample = {'flow': flow, 'meta': meta}
        if self.with_gt:
            sample['seg'] = seg

        return sample


class FlowLoader_Consistency(data.Dataset):
    def __init__(self, root_dir, resolution, split='train', year=2016,
                 to_rgb=False, with_gt=False):
        super(FlowLoader_Consistency, self).__init__()

        self.root_dir = root_dir
        self.resolution = resolution
        self.split = split
        self.year = year
        self.to_rgb = to_rgb
        self.with_gt = with_gt

        if isinstance(year, int):
            year = str(year)

        with open(os.path.join(root_dir, 'ImageSets', year, '{}.txt'.format(split))) as f:
            seqs = f.readlines()

        flo1_list = []
        flo2_list = []
        seg_list = []

        for seq in seqs:
            seq = seq.strip()

            flo1_files = sorted(os.listdir(os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq)))
            flo1_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_1_FlowFormer/Full-Resolution', seq, x), flo1_files))
            flo1_list.extend(flo1_paths)

            flo2_files = sorted(os.listdir(os.path.join(root_dir, 'Flows_gap_r1_FlowFormer/Full-Resolution', seq)))
            flo2_paths = list(map(lambda x: os.path.join(root_dir, 'Flows_gap_r1_FlowFormer/Full-Resolution', seq, x), flo2_files))
            flo2_list.extend(flo2_paths)

            seg_files = sorted(os.listdir(os.path.join(root_dir, 'Annotations_unsupervised/480p', seq)))[:-1]
            seg_paths = list(map(lambda x: os.path.join(root_dir, 'Annotations_unsupervised/480p', seq, x), seg_files))
            seg_list.extend(seg_paths)

        assert len(flo1_list) == len(flo2_list) == len(seg_list)
        self.flo1_list = flo1_list
        self.flo2_list = flo2_list
        self.seg_list = seg_list
        print('Done Initializing {0} set of DAVIS {1} Dataset.'.format(split, year))

    def __len__(self):
        return len(self.flo1_list)

    def __getitem__(self, index):
        flo1_file = self.flo1_list[index]
        flo2_file = self.flo2_list[index]
        seg_file = self.seg_list[index]

        img_path = os.path.splitext(seg_file)[0]
        img_path, img_name = os.path.split(img_path)
        seq_name = os.path.split(img_path)[-1]

        flow1 = helpers.readFlow(flo1_file, self.resolution, self.to_rgb)
        flo1 = helpers.readFlow(flo1_file, self.resolution, to_rgb=False)
        flow2 = helpers.readFlow(flo2_file, self.resolution, self.to_rgb)
        flo2 = helpers.readFlow(flo2_file, self.resolution, to_rgb=False)

        seg = helpers.readSeg(seg_file)
        ori_size = seg.shape
        if self.resolution is not None:
            if seg.shape != self.resolution:
                seg = cv2.resize(seg, (self.resolution[1], self.resolution[0]),
                                 interpolation=cv2.INTER_NEAREST)

        flow1 = torch.from_numpy(flow1).float()
        flow2 = torch.from_numpy(flow2).float()
        # flow = torch.stack([flow1, flow2], dim=0)
        seg = torch.from_numpy(seg).long()

        seg[seg == 255] = 0
        if self.year == '2016':
            seg[seg != 0] = 1
        seg = seg.float()
        seg = einops.rearrange(seg, 'h w -> (h w)')

        meta = {'seq_name': seq_name, 'img_name': img_name, 'ori_size': ori_size}
        sample = {'flow1': flow1, 'flow2': flow2, 'flo1': flo1, 'flo2': flo2, 'meta': meta}
        if self.with_gt:
            sample['seg'] = seg

        return sample
