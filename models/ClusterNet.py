import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.parameter import Parameter

from models.AE import AE, AE_NS
from models.modules import distributed_sinkhorn, l2_normalize


class ClusterNet(nn.Module):
    def __init__(self, arch, n_input, n_z, n_clusters):
        super(ClusterNet, self).__init__()
        self.n_clusters = n_clusters
        self.n_z = n_z

        self.auto_encoder = AE(arch, n_input, n_z)

    def cluster(self, z, n_iters=3):
        b = z.shape[0]
        self.prototypes = torch.randn((b, self.n_clusters, self.n_z), device=z.device, requires_grad=False)
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        for _ in range(n_iters):
            s = torch.einsum('bnd, bkd->bnk', l2_normalize(z), self.prototypes)
            s = torch.softmax(s, dim=-1)
            q_and_indexes = list(map(lambda x: distributed_sinkhorn(x[0]), torch.split(s, 1)))
            logits = torch.stack([q_and_indexes[i][0] for i in range(b)], dim=0)
            indexes = torch.stack([q_and_indexes[i][1] for i in range(b)], dim=0)
            one_hot = F.one_hot(indexes, num_classes=logits.shape[2]).float()
            prototypes = torch.einsum('bnk,bnd->bkd', one_hot, z) / torch.sum(one_hot, dim=1)[..., None]
            prototypes = torch.nan_to_num(prototypes)
            self.prototypes.data.copy_(l2_normalize(prototypes))

        # logits = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.prototypes)
        # indexes = F.softmax(logits, dim=-1)
        # indexes = torch.argmax(indexes, dim=-1)
        return logits, indexes

    def forward(self, x, pretrain_process=False):
        x_bar, z = self.auto_encoder(x)
        h, w = z.shape[2:]
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        preds = None
        if not pretrain_process:
            logits, indexes = self.cluster(z)
            logits = einops.rearrange(logits, 'b (h w) c -> b c h w', h=h, w=w)
            indexes = einops.rearrange(indexes, 'b (h w) -> b h w', h=h, w=w)
            z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
            preds = {'logits': logits, 'indexes': indexes}
            return x_bar, preds
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        return x_bar


class ClusterNet_BGC(nn.Module):
    def __init__(self, arch, n_input, n_z, n_clusters):
        super(ClusterNet_BGC, self).__init__()
        self.n_clusters = n_clusters
        self.n_z = n_z

        self.auto_encoder = AE(arch, n_input, n_z)

    def init_boundaryBox(self, h, w, duration=3):
        boundaryBox = torch.zeros(h - duration * 2, w - duration * 2)
        boundaryBox = F.pad(boundaryBox, (duration, duration, duration, duration), "constant", 1)
        num_pixel = torch.sum(boundaryBox)
        boundaryBox = einops.rearrange(boundaryBox, 'h w -> (h w)')
        return boundaryBox, num_pixel

    def calculate_saliency(self, z, z_size):
        # calculate background prototype
        b = z.shape[0]
        self.bg_prototype = torch.randn((b, 1, self.n_z), device=z.device, requires_grad=False)
        boundaryBox, num_pixel = self.init_boundaryBox(z_size[0], z_size[1])
        boundaryBox = boundaryBox.to(z.device)
        num_pixel = num_pixel.to(z.device)
        self.bg_prototype = torch.sum(z * boundaryBox[None, :, None], dim=1)[:, None, :] / num_pixel
        self.bg_prototype.data.copy_(l2_normalize(self.bg_prototype))
        bg_similarity = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.bg_prototype)
        mask = (bg_similarity - torch.min(bg_similarity, dim=1, keepdim=True)[0]
                ) / (torch.max(bg_similarity, dim=1, keepdim=True)[0] - torch.min(bg_similarity, dim=1, keepdim=True)[0])
        mask = 1.0 - mask
        return mask

    def cluster(self, z, n_iters=3):
        b = z.shape[0]
        self.prototypes = torch.randn((b, self.n_clusters, self.n_z), device=z.device, requires_grad=False)
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        for _ in range(n_iters):
            s = torch.einsum('bnd, bkd->bnk', l2_normalize(z), self.prototypes)
            s = torch.softmax(s, dim=-1)
            q_and_indexes = list(map(lambda x: distributed_sinkhorn(x[0]), torch.split(s, 1)))
            logits = torch.stack([q_and_indexes[i][0] for i in range(b)], dim=0)
            indexes = torch.stack([q_and_indexes[i][1] for i in range(b)], dim=0)
            one_hot = F.one_hot(indexes, num_classes=logits.shape[2]).float()
            prototypes = torch.einsum('bnk,bnd->bkd', one_hot, z) / torch.sum(one_hot, dim=1)[..., None]
            prototypes = torch.nan_to_num(prototypes)
            self.prototypes.data.copy_(l2_normalize(prototypes))

        # logits = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.prototypes)
        # indexes = F.softmax(logits, dim=-1)
        # indexes = torch.argmax(indexes, dim=-1)
        return logits, indexes

    def forward(self, x, pretrain_process=False):
        x_bar, z = self.auto_encoder(x)
        h, w = z.shape[2:]
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        mask = self.calculate_saliency(z, (h, w))
        preds = None
        if not pretrain_process:
            logits, indexes = self.cluster(z)
            logits = einops.rearrange(logits, 'b (h w) c -> b c h w', h=h, w=w)
            indexes = einops.rearrange(indexes, 'b (h w) -> b h w', h=h, w=w)
            z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
            preds = {'logits': logits, 'indexes': indexes}
            return x_bar, z, mask, preds
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        return x_bar, z, mask


class ClusterNet_BGC_NS(nn.Module):
    def __init__(self, arch, n_input, n_z, n_clusters):
        super(ClusterNet_BGC_NS, self).__init__()
        self.n_clusters = n_clusters
        self.n_z = n_z

        self.auto_encoder = AE_NS(arch, n_input, n_z)

    def init_boundaryBox(self, h, w, duration=3):
        boundaryBox = torch.zeros(h - duration * 2, w - duration * 2)
        boundaryBox = F.pad(boundaryBox, (duration, duration, duration, duration), "constant", 1)
        num_pixel = torch.sum(boundaryBox)
        boundaryBox = einops.rearrange(boundaryBox, 'h w -> (h w)')
        return boundaryBox, num_pixel

    def calculate_saliency(self, z, z_size):
        # calculate background prototype
        b = z.shape[0]
        self.bg_prototype = torch.randn((b, 1, self.n_z), device=z.device, requires_grad=False)
        boundaryBox, num_pixel = self.init_boundaryBox(z_size[0], z_size[1])
        boundaryBox = boundaryBox.to(z.device)
        num_pixel = num_pixel.to(z.device)
        self.bg_prototype = torch.sum(z * boundaryBox[None, :, None], dim=1)[:, None, :] / num_pixel
        self.bg_prototype.data.copy_(l2_normalize(self.bg_prototype))
        bg_similarity = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.bg_prototype)
        mask = (bg_similarity - torch.min(bg_similarity, dim=1, keepdim=True)[0]
                ) / (torch.max(bg_similarity, dim=1, keepdim=True)[0] - torch.min(bg_similarity, dim=1, keepdim=True)[0])
        mask = 1.0 - mask
        return mask

    def cluster(self, z, n_iters=3):
        b = z.shape[0]
        self.prototypes = torch.randn((b, self.n_clusters, self.n_z), device=z.device, requires_grad=False)
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        for _ in range(n_iters):
            s = torch.einsum('bnd, bkd->bnk', l2_normalize(z), self.prototypes)
            s = torch.softmax(s, dim=-1)
            q_and_indexes = list(map(lambda x: distributed_sinkhorn(x[0]), torch.split(s, 1)))
            logits = torch.stack([q_and_indexes[i][0] for i in range(b)], dim=0)
            indexes = torch.stack([q_and_indexes[i][1] for i in range(b)], dim=0)
            one_hot = F.one_hot(indexes, num_classes=logits.shape[2]).float()
            prototypes = torch.einsum('bnk,bnd->bkd', one_hot, z) / torch.sum(one_hot, dim=1)[..., None]
            prototypes = torch.nan_to_num(prototypes)
            self.prototypes.data.copy_(l2_normalize(prototypes))

        # logits = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.prototypes)
        # indexes = F.softmax(logits, dim=-1)
        # indexes = torch.argmax(indexes, dim=-1)
        return logits, indexes

    def forward(self, x, pretrain_process=False):
        x_bar, z = self.auto_encoder(x)
        h, w = z.shape[2:]
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        mask = self.calculate_saliency(z, (h, w))
        preds = None
        if not pretrain_process:
            logits, indexes = self.cluster(z)
            logits = einops.rearrange(logits, 'b (h w) c -> b c h w', h=h, w=w)
            indexes = einops.rearrange(indexes, 'b (h w) -> b h w', h=h, w=w)
            z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
            preds = {'logits': logits, 'indexes': indexes}
            return x_bar, z, mask, preds
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        return x_bar, z, mask


class ClusterNet_Contrast(nn.Module):
    def __init__(self, arch, n_input, n_z, n_clusters):
        super(ClusterNet_Contrast, self).__init__()
        self.n_clusters = n_clusters
        self.n_z = n_z

        self.auto_encoder = AE(arch, n_input, n_z)

    def cluster(self, z, n_iters=3):
        b = z.shape[0]
        self.prototypes = torch.randn((b, self.n_clusters, self.n_z), device=z.device, requires_grad=False)
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        for _ in range(n_iters):
            s = torch.einsum('bnd, bkd->bnk', l2_normalize(z), self.prototypes)
            s = torch.softmax(s, dim=-1)
            q_and_indexes = list(map(lambda x: distributed_sinkhorn(x[0]), torch.split(s, 1)))
            logits = torch.stack([q_and_indexes[i][0] for i in range(b)], dim=0)
            indexes = torch.stack([q_and_indexes[i][1] for i in range(b)], dim=0)
            one_hot = F.one_hot(indexes, num_classes=logits.shape[2]).float()
            prototypes = torch.einsum('bnk,bnd->bkd', one_hot, z) / torch.sum(one_hot, dim=1)[..., None]
            prototypes = torch.nan_to_num(prototypes)
            self.prototypes.data.copy_(l2_normalize(prototypes))

        # logits = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.prototypes)
        # indexes = F.softmax(logits, dim=-1)
        # indexes = torch.argmax(indexes, dim=-1)
        return logits, indexes

    def forward(self, x):
        x_bar, z = self.auto_encoder(x)
        h, w = z.shape[2:]
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        logits, indexes = self.cluster(z)
        logits = einops.rearrange(logits, 'b (h w) c -> b c h w', h=h, w=w)
        indexes = einops.rearrange(indexes, 'b (h w) -> b h w', h=h, w=w)
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        preds = {'logits': logits, 'indexes': indexes}
        return x_bar, z, preds, self.prototypes


class ClusterNet_BGC_Contrast(nn.Module):
    def __init__(self, arch, n_input, n_z, n_clusters):
        super(ClusterNet_BGC_Contrast, self).__init__()
        self.n_clusters = n_clusters
        self.n_z = n_z

        self.auto_encoder = AE(arch, n_input, n_z)

    def init_boundaryBox(self, h, w, duration=3):
        boundaryBox = torch.zeros(h - duration * 2, w - duration * 2)
        boundaryBox = F.pad(boundaryBox, (duration, duration, duration, duration), "constant", 1)
        num_pixel = torch.sum(boundaryBox)
        boundaryBox = einops.rearrange(boundaryBox, 'h w -> (h w)')
        return boundaryBox, num_pixel

    def calculate_saliency(self, z, z_size):
        # calculate background prototype
        b = z.shape[0]
        self.bg_prototype = torch.randn((b, 1, self.n_z), device=z.device, requires_grad=False)
        boundaryBox, num_pixel = self.init_boundaryBox(z_size[0], z_size[1])
        boundaryBox = boundaryBox.to(z.device)
        num_pixel = num_pixel.to(z.device)
        self.bg_prototype = torch.sum(z * boundaryBox[None, :, None], dim=1)[:, None, :] / num_pixel
        self.bg_prototype.data.copy_(l2_normalize(self.bg_prototype))
        bg_similarity = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.bg_prototype)
        mask = (bg_similarity - torch.min(bg_similarity, dim=1, keepdim=True)[0]
                ) / (torch.max(bg_similarity, dim=1, keepdim=True)[0] - torch.min(bg_similarity, dim=1, keepdim=True)[0])
        mask = 1.0 - mask
        return mask

    def cluster(self, z, n_iters=3):
        b = z.shape[0]
        self.prototypes = torch.randn((b, self.n_clusters, self.n_z), device=z.device, requires_grad=False)
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        for _ in range(n_iters):
            s = torch.einsum('bnd, bkd->bnk', l2_normalize(z), self.prototypes)
            s = torch.softmax(s, dim=-1)
            q_and_indexes = list(map(lambda x: distributed_sinkhorn(x[0]), torch.split(s, 1)))
            logits = torch.stack([q_and_indexes[i][0] for i in range(b)], dim=0)
            indexes = torch.stack([q_and_indexes[i][1] for i in range(b)], dim=0)
            one_hot = F.one_hot(indexes, num_classes=logits.shape[2]).float()
            prototypes = torch.einsum('bnk,bnd->bkd', one_hot, z) / torch.sum(one_hot, dim=1)[..., None]
            prototypes = torch.nan_to_num(prototypes)
            self.prototypes.data.copy_(l2_normalize(prototypes))

        # logits = torch.einsum('bnd,bkd->bnk', l2_normalize(z), self.prototypes)
        # indexes = F.softmax(logits, dim=-1)
        # indexes = torch.argmax(indexes, dim=-1)
        return logits, indexes

    def forward(self, x):
        x_bar, z = self.auto_encoder(x)
        h, w = z.shape[2:]
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        mask = self.calculate_saliency(z, (h, w))
        logits, indexes = self.cluster(z)
        logits = einops.rearrange(logits, 'b (h w) c -> b c h w', h=h, w=w)
        indexes = einops.rearrange(indexes, 'b (h w) -> b h w', h=h, w=w)
        z = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        preds = {'logits': logits, 'indexes': indexes}
        return x_bar, z, mask, preds, self.prototypes
