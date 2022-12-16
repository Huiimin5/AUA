import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

from . import ramps

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
def dice_loss(score, target, reduction = True):
    score = score.float()
    target = target.float()
    smooth = 1e-5
    if reduction:
        intersect = torch.sum(score * target)
    else:
        intersect =  score * target
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
def class_mean_dice_loss_sample_wise(score, target):
    score = score.float()
    target = target.float()
    num_of_class = score.size()[0]
    smooth = 1e-5
    lc = 0
    for c in range(num_of_class):
        intersect = torch.sum(score[c] * target[c])
        y_sum = torch.sum(target[c] * target[c])
        z_sum = torch.sum(score[c] * score[c])
        lc += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - lc / num_of_class
    return loss
def class_mean_dice_loss_batch_wise(score, target):
    score = score.float()
    target = target.float()
    num_of_class = score.size()[1]
    smooth = 1e-5
    lc = 0
    for c in range(num_of_class):
        intersect = torch.sum(score[:,c] * target[:,c])
        y_sum = torch.sum(target[:,c] * target[:,c])
        z_sum = torch.sum(score[:,c] * score[:,c])
        lc += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - lc / num_of_class
    return loss

def mse_loss(score, target):
    score = score.float()
    target = target.float()
    mse_loss = torch.sum((score - target) ** 2)
    return mse_loss
def kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='sum')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def JSD_loss(net_1_logits, net_2_logits):
    # bs, nc, sx, sy, sz = net_1_logits.size()
    # net_1_logits = net_1_logits.permute(0,2,3,4,1).contiguous().view((bs*sx*sy*sz, nc))
    # net_2_logits = net_2_logits.permute(0, 2, 3, 4, 1).contiguous().view((bs * sx * sy * sz, nc))
    net_1_probs = F.softmax(net_1_logits, dim=1)
    net_2_probs = F.softmax(net_2_logits, dim=1)

    total_m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m)
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m)

    return (0.5 * loss)

def dice_coefficient(score, target):
    score = score.float()
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss
def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

import torch.nn.functional as F
import torch.nn as nn
import math

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, logits: torch.tensor, target: torch.tensor, **kwargs):
        return super().forward(logits, target)


def fixed_re_parametrization_trick(dist, num_samples):
    assert num_samples % 2 == 0
    samples = dist.rsample((num_samples // 2,))
    mean = dist.mean.unsqueeze(0)
    samples = samples - mean
    return torch.cat([samples, -samples]) + mean

def loss_mc_integral_given_samples(logit_sample, target, num_mc_samples):
    batch_size = target.size()[0]
    num_classes = logit_sample.size()[2]
    target = target.unsqueeze(1)
    target = target.expand((num_mc_samples,) + target.shape)

    flat_size = num_mc_samples * batch_size
    logit_sample = logit_sample.contiguous().view((flat_size, num_classes, -1))
    target = target.reshape((flat_size, -1))

    log_prob = -F.cross_entropy(logit_sample, target, reduction='none').view((num_mc_samples, batch_size, -1))
    loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(num_mc_samples))
    loss = -loglikelihood
    return loss


def generalised_energy_distance(sample_arr, gt_arr, pairwise_dist_metric='dice', weight_div1 = 1, weight_div2=1):
    """
    :param sample_arr: expected shape S, H, W, D
    :param gt_arr: S, H, W, D
    :return:
    """

    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    if pairwise_dist_metric == 'dice':
        criteria = dice_loss
    elif pairwise_dist_metric == 'kl':
        criteria = kl_loss
    elif pairwise_dist_metric == 'mse':
        criteria = symmetric_mse_loss
    elif pairwise_dist_metric == 'cls_mean_dice':
        criteria = class_mean_dice_loss_sample_wise


    for i in range(N):
        for j in range(M):
            d_sy.append(criteria(sample_arr[i, ...], gt_arr[j, ...]))
        for j in range(N):
            d_ss.append(criteria(sample_arr[i, ...], sample_arr[j, ...]))

    for i in range(M):
        for j in range(M):
            # print(dist_fct(gt_arr[i,...], gt_arr[j,...]))
            d_yy.append(criteria(gt_arr[i, ...], gt_arr[j, ...]))
    diversity1 = (1. / N ** 2) * sum(d_ss)
    diversity2 = (1. / M ** 2) * sum(d_yy)
    if weight_div1 == 1 and weight_div2 == 1:
        return (2. / (N * M)) * sum(d_sy) - diversity1 - diversity2, diversity1, diversity2
    else:
        return (1. / (N * M)) * sum(d_sy) - weight_div1 * diversity1 - weight_div2 * diversity2, diversity1, diversity2

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class PixelContrastiveLoss(nn.Module):
    def __init__(self, tau, IGNORE_LABEL=255):
        super(PixelContrastiveLoss, self).__init__()
        self.tau = tau
        self.IGNORE_LABEL = IGNORE_LABEL

    def forward(self, Mean, CoVariance, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Mean: shape: (C, A) the mean representation of each class
            CoVariance: shape: (C, A) the diagonals of covariance matrices,
                        i.e., the variance  of each dimension of the features for each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )

        Returns:

        """
        assert not Mean.requires_grad
        assert not CoVariance.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1

        # remove IGNORE_LABEL pixels
        mask = (labels != self.IGNORE_LABEL)
        labels = labels[mask]
        feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)
        Mean = F.normalize(Mean, p=2, dim=1)
        CoVariance = F.normalize(CoVariance, p=2, dim=1)

        temp1 = feat.mm(Mean.permute(1, 0).contiguous())
        CoVariance = CoVariance / self.tau
        temp2 = 0.5 * feat.pow(2).mm(CoVariance.permute(1, 0).contiguous())

        logits = temp1 + temp2
        logits = logits / self.tau

        ce_criterion = nn.CrossEntropyLoss()
        ce_loss = ce_criterion(logits, labels)
        pcl_loss = 0.5 * torch.sum(feat.pow(2).mul(CoVariance[labels]), dim=1).mean() / self.tau

        loss = ce_loss + pcl_loss
        return loss
