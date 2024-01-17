# from itertools import combinations_with_replacement
import einops

import torch
import torch.nn as nn
# import torch.nn.functional as F

from models.modules import l2_normalize
# distributed_sinkhorn
# from utils.helpers import inverse_flow_warp


class BGCLoss(nn.Module):
    def __init__(self, reduction='mean', temperature=0.1):
        super(BGCLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, z, mask):
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        fg = torch.sum(z * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        bg = torch.sum(z * ~mask, dim=1, keepdim=True) / torch.sum(~mask, dim=1, keepdim=True)
        prototypes = torch.cat([bg, fg], dim=1)
        all_pairs = torch.exp(torch.einsum('bkc,bnc->bkn', l2_normalize(prototypes), l2_normalize(z)) / self.temperature)
        pos_pairs = torch.gather(all_pairs, 1, mask.permute(0, 2, 1).long())
        all_pairs = torch.sum(all_pairs, dim=1, keepdim=True)
        loss = -1 * torch.mean(torch.log(pos_pairs / all_pairs))
        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, reduction='mean', temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

        self.ce_criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, z, s, prototypes):
        b, c, h, w = z.shape
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        s = einops.rearrange(s, 'b h w -> b (h w)')
        logits = torch.einsum('bkc,bnc->bkn', prototypes, l2_normalize(z)) / self.temperature

        loss = self.ce_criterion(logits, s.long())

        return loss


class InterClassLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(InterClassLoss, self).__init__()

        self.reduction = reduction

    def forward(self, z, s, prototypes):
        b, c, h, w = z.shape
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        s = einops.rearrange(s, 'b h w -> b (h w)')
        cosine_similarity = torch.einsum('bkc,bnc->bkn', prototypes, l2_normalize(z))
        logits = torch.gather(cosine_similarity, 1, s[:, None, :])
        loss = (1 - logits).pow(2).mean()

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ClusterLoss, self).__init__()

        self.mse_criterion = nn.MSELoss(reduction=reduction)

    def forward(self, x_bar, x):
        mse_loss = self.mse_criterion(x_bar, x)

        return mse_loss


class ClusterLoss_BGC(nn.Module):
    def __init__(self, reduction='mean', weights=0.1, threshold=0.5):
        super(ClusterLoss_BGC, self).__init__()
        self.weights = weights
        self.threshold = threshold

        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.mask_criterion = BGCLoss(reduction=reduction)

    def forward(self, x_bar, x, z, mask):
        mask = mask >= self.threshold
        mse_loss = self.mse_criterion(x_bar, x)
        mask_loss = self.mask_criterion(z, mask)

        loss = mse_loss + self.weights * mask_loss

        return loss


class ClusterLoss_Contrast(nn.Module):
    def __init__(self, mode='A', reduction='mean', weights=[1, 0.01, 0.01]):
        super(ClusterLoss_Contrast, self).__init__()

        self.mode = mode
        self.reduction = reduction
        self.weights = weights
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.infoNCE_criterion = InfoNCELoss(reduction=reduction)
        self.interClass_criterion = InterClassLoss(reduction=reduction)

    def forward(self, x_bar, x, z, s, prototypes):
        mse_loss = self.mse_criterion(x_bar, x)
        infoNCE_loss = self.infoNCE_criterion(z, s, prototypes)
        interClass_loss = self.interClass_criterion(z, s, prototypes)
        if self.mode == 'A':
            loss = self.weights[0] * mse_loss + self.weights[1] * infoNCE_loss
        elif self.mode == 'B':
            loss = self.weights[0] * mse_loss + self.weights[2] * interClass_loss
        elif self.mode == 'C':
            loss = self.weights[0] * mse_loss + self.weights[1] * infoNCE_loss + self.weights[2] * interClass_loss
        else:
            raise NotImplementedError('{} mode is not implemented.'.format(self.mode))

        return loss


class ClusterLoss_BGC_Contrast(nn.Module):
    def __init__(self, mode='A', reduction='mean', weights=[1, 0.01, 0.01, 0.01], threshold=0.5):
        super(ClusterLoss_BGC_Contrast, self).__init__()

        self.mode = mode
        self.reduction = reduction
        self.weights = weights
        self.threshold = threshold
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.mask_criterion = BGCLoss(reduction=reduction)
        self.infoNCE_criterion = InfoNCELoss(reduction=reduction)
        self.interClass_criterion = InterClassLoss(reduction=reduction)

    def forward(self, x_bar, x, z, mask, s, prototypes):
        mask = mask >= self.threshold
        mse_loss = self.mse_criterion(x_bar, x)
        mask_loss = self.mask_criterion(z, mask)
        infoNCE_loss = self.infoNCE_criterion(z, s, prototypes)
        interClass_loss = self.interClass_criterion(z, s, prototypes)
        if self.mode == 'A':
            loss = self.weights[0] * mse_loss + self.weights[1] * mask_loss + self.weights[2] * infoNCE_loss
        elif self.mode == 'B':
            loss = self.weights[0] * mse_loss + self.weights[1] * mask_loss + self.weights[3] * interClass_loss
        elif self.mode == 'C':
            loss = self.weights[0] * mse_loss + self.weights[1] * mask_loss + \
                self.weights[2] * infoNCE_loss + \
                self.weights[3] * interClass_loss
        else:
            raise NotImplementedError('{} mode is not implemented.'.format(self.mode))

        return loss
