import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.stochastic_vnet import StochasticVNetSupCon, StochasticVNetSupCon_noout
from dataloaders import utils
from utils import ramps, losses
import torchio
from dataloaders.pancreas import PancreasMasked as Pancreas
from dataloaders.la_heart import RandomCropMasked as RandomCrop, \
    CenterCropMasked as CenterCrop, RandomRotFlipMasked as RandomRotFlip, RandomFlipMasked as RandomFlip,\
    RandomAffineMasked as RandomAffine, GaussianBlurMasked as GaussianBlur,\
    NoAugMasked as NoAug,   \
    ToTensorMasked as ToTensor, LabeledBatchSampler, UnlabeledBatchSampler
from val_3D import test_batch
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
# todo: adapt from train_LA_3d_ged_seperate_aug.py
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_3items/Pancreas-CT-training', help='Name of Experiment')
parser.add_argument('--image_list_path', type=str, default='pancreas_train.list', help='image_list_path')

parser.add_argument('--exp', type=str,  default='pancreas_exp_000', help='model_name')
parser.add_argument('--labeled_num', type=int,  default=12, help='labeled_num')
parser.add_argument('--total_num', type=int,  default=60, help='total_num')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### todo: parameters for supervised contrastive loss
parser.add_argument('--with_contrastive_loss', type=str2bool,  default=True, help='with_contrastive_loss')
parser.add_argument('--cross_image_contrast', type=str2bool,  default=True, help='cross_image_contrast')
parser.add_argument('--baseline_noout', type=str2bool,  default=True, help='baseline_noout')
parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
parser.add_argument('--contrast_mode', type=str, default='one', help='contrast_mode') # ['one', 'all']
parser.add_argument('--contrast_pixel_sampling', type=str, default='near_boundary', help='contrast_pixel_sampling') # ['near_boundary', 'uncertainty']
parser.add_argument('--sup_cont_weight', type=float, default=1, help='sup_cont_weight')
parser.add_argument('--near_boundary_range', type=int,  default=1, help='near_boundary_range')
parser.add_argument('--random_sampled_num', type=int,  default=1000, help='random_sampled_num')
parser.add_argument('--sup_cont_rampdown', type=float,  default=40, help='sup_cont_rampdown')
parser.add_argument('--sup_cont_rampdown_scheme', type=str,  default='None', help='sup_cont_rampdown_scheme')
parser.add_argument('--oversample_ratio', type=int,  default=3, help='oversample_ratio')
parser.add_argument('--importance_sample_ratio', type=float,  default=0.75, help='importance_sample_ratio')
parser.add_argument('--cross_image_sampling', type=str2bool,  default=False, help='cross_image_sampling')
parser.add_argument('--head_normalization', type=str,  default='none', help='head_normalization')
parser.add_argument('--head_layer_num', type=int,  default=3, help='head_layer_num')
## parameters for pseudo contrastive loss
parser.add_argument('--with_unsup_contrastive_loss', type=str2bool,  default=False, help='with_unsup_contrastive_loss')
parser.add_argument('--cross_unsup_image_contrast', type=str2bool,  default=True, help='cross_unsup_image_contrast')
parser.add_argument('--cross_unsup_image_sampling', type=str2bool,  default=False, help='cross_unsup_image_sampling')
parser.add_argument('--unsup_cont_weight', type=float, default=1, help='unsup_cont_weight')
parser.add_argument('--unsup_cont_rampup_scheme', type=str,  default='sigmoid_rampup', help='unsup_cont_rampup_scheme')
parser.add_argument('--unsup_cont_rampup', type=float,  default=40, help='sup_cont_rampdown')
parser.add_argument('--random_unsup_sampled_num', type=int,  default=1000, help='random_unsup_sampled_num')

## parameters for unsup contrast using subvolume after decoder
parser.add_argument('--with_unsup_contrastive_subvolume_after_decoder', type=str2bool,  default=False, help='with_unsup_contrastive_subvolume_after_decoder')
parser.add_argument('--unsup_average_size', type=int, default=4, help='unsup_cont_weight')
parser.add_argument('--subvolume_after_decoder_cross_image_contrast', type=str2bool,  default=False, help='subvolume_after_decoder_cross_image_contrast')
parser.add_argument('--unsup_subvolume_after_decoder_cont_weight', type=float, default=0.1, help='unsup_subvolume_after_decoder_cont_weight')
parser.add_argument('--unsup_subvolume_after_decoder_cont_rampup_scheme', type=str,  default='sigmoid_rampup', help='unsup_subvolume_after_decoder_cont_rampup_scheme')
parser.add_argument('--unsup_subvolume_after_decoder_rampup', type=float,  default=40, help='unsup_subvolume_after_decoder_rampup')

### costs
parser.add_argument('--ab_ce', type=str2bool,  default=False, help='ab_ce')
parser.add_argument('--ab_ce_ramp_type', type=str,  default='log_rampup', help='ab_ce_ramp_type')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="ged", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--num_mc_samples', type=int,  default=20, help='num_mc_samples')
parser.add_argument('--with_dice', type=str2bool,  default=True, help='with_dice loss')
parser.add_argument('--perturbation_weight_ema', type=float,  default=1.0, help='perturbation weight ema')
parser.add_argument('--perturbation_weight_feature_ema', type=float,  default=0, help='perturbation feature weight ema')
parser.add_argument('--num_feature_perturbated', type=int,  default=1, help='num_feature_perturbated')
parser.add_argument('--uniform_range', type=float,  default=0.3, help='uniform range')
parser.add_argument('--uncertainty_consistency', type=float,  default=0, help='uncertainty_consistency')
parser.add_argument('--ssn_rank', type=int,  default=10, help='ssn rank')
parser.add_argument('--pairwise_dist_metric', type=str,  default='dice', help='ssn rank')
parser.add_argument('--oracle_checking', type=str2bool,  default=False, help='oracle_checking')
parser.add_argument('--with_uncertainty_mask', type=str2bool,  default=False, help='with_uncertainty_mask')
# ablation study parameters for uncertianty mask
parser.add_argument('--cov_diag_consistency_weight', type=float,  default=1, help='cov_diag_consistency_weight')
parser.add_argument('--cov_factor_consistency_weight', type=float,  default=1, help='cov_factor_consistency_weight')


# ablation study parameters for generalized energy distance
parser.add_argument('--weight_div1', type=float,  default=1, help='weight for diversity1 in loss function')
parser.add_argument('--weight_div2', type=float,  default=1, help='weight for diversity2 in loss function')
# region level generalized energy distance
parser.add_argument('--region_level', type=str2bool,  default=False, help='region level generalized energy distance')
parser.add_argument('--num_of_region_h', type=int,  default=1, help='num_of_region_h')
parser.add_argument('--num_of_region_w', type=int,  default=1, help='num_of_region_w')
parser.add_argument('--num_of_region_d', type=int,  default=1, help='num_of_region_d')
# ablation study of energy based distance
parser.add_argument('--energy_based', type=str2bool,  default=True, help='oracle_checking')
# self information consistency
parser.add_argument('--lambda_g', type=float,  default=0, help='lambda_g')
# energy based self information consistency
parser.add_argument('--energy_g', type=str2bool,  default=False, help='energy_g')
# ablation study of ablation study
parser.add_argument('--bn_type', type=str,  default='batchnorm', help='bn_type')
# ablation study of mask for ssn
parser.add_argument('--with_mask_ssn', type=str2bool,  default=True, help='with_mask_ssn')
# ablationn study of data loader augmentation
parser.add_argument('--data_loader_labeled_aug', type=str, default='RandomRotFlip', help='data_loader_labeled_aug')
parser.add_argument('--unlabeled_aug_with_resize', type=str2bool, default=False, help='unlabeled_aug_with_resize')
parser.add_argument('--unlabeled_aug_with_gaussian_blur', type=str2bool, default=False, help='unlabeled_aug_with_gaussian_blur')
parser.add_argument('--unlabeled_aug_with_rotationflip', type=str2bool, default=False, help='unlabeled_aug_with_rotationflip')
parser.add_argument('--unlabeled_aug_with_flip', type=str2bool, default=False, help='unlabeled_aug_with_flip')
parser.add_argument('--unlabeled_augT_with_gaussian_blur', type=str2bool, default=False, help='unlabeled_augT_with_gaussian_blur')

# for debug
parser.add_argument('--transform_fixed', type=str2bool, default=False, help='')
parser.add_argument('--sampler_fixed', type=str2bool, default=False, help='')
# ablation study of ramp up scheduler
parser.add_argument('--ramp_up_scheduler', type=str, default='sigmoid_rampup', help='ramp_up_scheduler')
# ablation study of supervised loss
parser.add_argument('--no_uncertainty_sup', type=str2bool, default=False, help='no_uncertainty_sup')
# try diff augmentation strategies to T input image
# try diff way to update teacher
parser.add_argument('--ema_dec_only', type=str2bool, default=False, help='ema_dec_only')
parser.add_argument('--ema_enc_only', type=str2bool, default=False, help='ema_enc_only')

# resume
parser.add_argument('--resume', type=str2bool, default=False, help='resume')
parser.add_argument('--load_epoch_num', type=int, default=3000, help='load_epoch_num') # or 60 for upper bound
parser.add_argument('--load_model_name', type=str, default='exp_pancreas_084', help='load_model_name') # or 60 for upper bound
parser.add_argument('--fix_bn_after_resume', type=str2bool, default=True, help='fix_bn_after_resume')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
ab_ce = args.ab_ce
with_dice = args.with_dice
num_mc_samples = args.num_mc_samples
perturbation_weight_ema = args.perturbation_weight_ema
ssn_rank = args.ssn_rank
pairwise_dist_metric = args.pairwise_dist_metric
oracle_checking = args.oracle_checking
with_uncertainty_mask = args.with_uncertainty_mask
perturbation_weight_feature_ema = args.perturbation_weight_feature_ema
uniform_range = args.uniform_range
num_feature_perturbated = args.num_feature_perturbated
uncertainty_consistency = args.uncertainty_consistency
# ablation study parameters for uncertainty estimation
cov_diag_consistency_weight = args.cov_diag_consistency_weight
cov_factor_consistency_weight = args.cov_factor_consistency_weight
# ablation study parameters for generalized energy distance
weight_div1 = args.weight_div1
weight_div2 = args.weight_div2
# region level generalized energy distance
region_level = args.region_level
num_of_region_h=args.num_of_region_h
num_of_region_w=args.num_of_region_w
num_of_region_d=args.num_of_region_d
# self information consistency
lambda_g = args.lambda_g
energy_g = args.energy_g
# ablation study of ablation study
bn_type = args.bn_type
with_mask_ssn = args.with_mask_ssn
# ablation study of data_loader_labeled_aug
data_loader_labeled_aug = args.data_loader_labeled_aug

unlabeled_aug_with_resize = args.unlabeled_aug_with_resize
unlabeled_aug_with_gaussian_blur = args.unlabeled_aug_with_gaussian_blur
unlabeled_aug_with_rotationflip = args.unlabeled_aug_with_rotationflip
unlabeled_aug_with_flip = args.unlabeled_aug_with_flip
unlabeled_augT_with_gaussian_blur = args.unlabeled_augT_with_gaussian_blur
# ablation study of ramp_up_scheduler
ramp_up_scheduler = args.ramp_up_scheduler
# ablation study of supervised loss
no_uncertainty_sup = args.no_uncertainty_sup

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)


def get_current_consistency_weight(epoch, consistency):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if ramp_up_scheduler == 'sigmoid_rampup':
        return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    elif ramp_up_scheduler == 'linear_rampup':
        return consistency * ramps.linear_rampup(epoch, args.consistency_rampup)
    elif ramp_up_scheduler == 'log_rampup':
        return consistency * ramps.log_rampup(epoch, args.consistency_rampup)
    elif ramp_up_scheduler == 'exp_rampup':
        return consistency * ramps.exp_rampup(epoch, args.consistency_rampup)

def get_sup_cont_weight(epoch, weight):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if args.sup_cont_rampdown_scheme == 'quadratic_rampdown':
        return weight * ramps.quadratic_rampdown(epoch, args.sup_cont_rampdown)
    elif args.sup_cont_rampdown_scheme == 'cosine_rampdown':
        return weight * ramps.cosine_rampdown(epoch, args.sup_cont_rampdown)
    else:
        return weight

def get_unsup_cont_weight(epoch, weight, scheme = args.unsup_cont_rampup_scheme, ramp_up = args.unsup_cont_rampup):
    if  scheme == 'sigmoid_rampup':
        return weight * ramps.sigmoid_rampup(epoch, ramp_up)
    elif scheme == 'linear_rampup':
        return weight * ramps.linear_rampup(epoch, ramp_up)
    elif scheme == 'log_rampup':
        return weight * ramps.log_rampup(epoch, ramp_up)
    elif scheme == 'exp_rampup':
        return weight * ramps.exp_rampup(epoch, ramp_up)
    else:
        return weight

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # print(ema_model.state_dict()['block_one.conv.1.running_var'])

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def update_ema_variables_decoder_only(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.named_parameters(), model.named_parameters()):
        ema_param_name, ema_param = ema_params
        param_name, param = params
        if ema_param_name.split('.')[0] in {'block_five', 'block_four', 'block_four_dw', 'block_three_dw',
                                            'block_three', 'block_two_dw', 'block_one_dw', 'block_two', 'block_one'}:
            ema_param.data.copy_(param.data)
        else:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def update_ema_variables_encoder_only(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.named_parameters(), model.named_parameters()):
        ema_param_name, ema_param = ema_params
        param_name, param = params
        if ema_param_name.split('.')[0] in {'block_five', 'block_four', 'block_four_dw', 'block_three_dw',
                                            'block_three', 'block_two_dw', 'block_one_dw', 'block_two', 'block_one'}:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        else:
            ema_param.data.copy_(param.data)


def get_supervised_loss_no_uncertainty(outputs, label_batch):
    loss_seg = F.cross_entropy(outputs, label_batch)
    outputs_soft = F.softmax(outputs, dim=1)
    loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
    supervised_loss = 0.5 * (loss_seg + loss_seg_dice)
    return supervised_loss, loss_seg, loss_seg_dice

# expected input: outputs[:labeled_bs], label_batch[:labeled_bs]
def get_superivsed_loss(output_logits, label, mc_samples, with_dice=True, ):
    b,c,h,w,d = output_logits.size()
    loss_seg_mc_intg = losses.loss_mc_integral_given_samples(mc_samples, label, num_mc_samples) / (h * w * d)
    output_prob = F.softmax(output_logits, dim=1)

    if with_dice:
        loss_seg_dice = losses.dice_loss(output_prob[:, 1, :, :, :], label == 1)
        supervised_loss = 0.5 * (loss_seg_mc_intg + loss_seg_dice)

    else:
        loss_seg_dice = torch.zeros([1]).cuda()
        supervised_loss = loss_seg_mc_intg + loss_seg_dice
    return supervised_loss, loss_seg_mc_intg, loss_seg_dice
def get_annealed_superivsed_loss(output_logits, label, mc_samples, iter_per_epoch, curr_iter, epoch,   ):
    b,c,h,w,d = output_logits.size()
    loss_seg_mc_intg, anneal_threshold = losses.loss_mc_integral_given_samples_annealed(mc_samples, label, num_mc_samples, iter_per_epoch, max_epoch,
                                                                      num_classes, curr_iter, epoch, args.ab_ce_ramp_type)
    loss_seg_mc_intg/= (h * w * d)
    supervised_loss = loss_seg_mc_intg
    return supervised_loss, loss_seg_mc_intg, 0, anneal_threshold
def oracle_check(output_logits, label, mc_samples ):
    b,c,h,w,d = output_logits.size()
    loss_seg_mc_intg = losses.loss_mc_integral_given_samples(mc_samples, label, num_mc_samples) / (h * w * d)
    output_prob = F.softmax(output_logits, dim=1)
    loss_seg_dice = losses.dice_loss(output_prob[:, 1, :, :, :], label == 1)
    supervised_loss = 0.5 * (loss_seg_mc_intg + loss_seg_dice)

    _, output = torch.max(output_logits, dim=1)
    dice_coefficient = losses.dice_loss(output, label)
    return supervised_loss, loss_seg_mc_intg, loss_seg_dice, dice_coefficient


# expected input: outputs[labeled_bs:], ema_output
def get_unsupervised_loss_before_0218(consistency_weight, stochastic_samples_logits, ema_stochastic_samples_logits):
    batch_sample_num = stochastic_samples_logits.size()[1]
    consistency_dist_each_batch_sample = []
    for batch_sample_id in range(batch_sample_num):
        if pairwise_dist_metric == 'dice':
            _, stochastic_samples_pred = torch.max(stochastic_samples_logits[:, batch_sample_id], dim=1)
            _, ema_stochastic_samples_pred = torch.max(ema_stochastic_samples_logits[:, batch_sample_id], dim=1)
            consistency_dist_each_batch_sample.append(
                losses.generalised_energy_distance(stochastic_samples_pred, ema_stochastic_samples_pred,pairwise_dist_metric)[0])
        elif pairwise_dist_metric == 'mse':
            consistency_dist_each_batch_sample.append(
                losses.generalised_energy_distance(stochastic_samples_logits[:, batch_sample_id], ema_stochastic_samples_logits[:, batch_sample_id],pairwise_dist_metric)[0])
    consistency_dist = torch.stack(consistency_dist_each_batch_sample).mean()
    consistency_loss = consistency_weight * consistency_dist
    return consistency_loss, consistency_dist

def get_unsupervised_loss(consistency_weight, stochastic_samples_logits, ema_stochastic_samples_logits):
    batch_sample_num = stochastic_samples_logits.size()[1]
    consistency_dist_each_batch_sample = []
    for batch_sample_id in range(batch_sample_num):
        # import pdb
        # pdb.set_trace()
        stochastic_samples_pred = F.softmax(stochastic_samples_logits[:, batch_sample_id], dim=1)
        ema_stochastic_samples_pred = F.softmax(ema_stochastic_samples_logits[:, batch_sample_id], dim=1)
        if pairwise_dist_metric == 'kl':
            # consistency_dist_each_batch_sample.append(
            #     losses.generalised_energy_distance(stochastic_samples_logits[:, batch_sample_id],
            #                                        ema_stochastic_samples_logits[:, batch_sample_id], pairwise_dist_metric)[0])
            if region_level:
                consistency_dist_each_batch_sample.append(
                    losses.generalised_energy_distance_region_wise(stochastic_samples_logits[:, batch_sample_id],
                                                                   ema_stochastic_samples_logits[:, batch_sample_id],
                                                                   pairwise_dist_metric, weight_div1, weight_div2,
                                                                   num_of_region_h, num_of_region_w, num_of_region_d)[0])
            else:
                consistency_dist_each_batch_sample.append(
                    losses.generalised_energy_distance(stochastic_samples_logits[:, batch_sample_id],
                                                       ema_stochastic_samples_logits[:, batch_sample_id],
                                                       pairwise_dist_metric, weight_div1, weight_div2,)[0])

        else:
            # consistency_dist_each_batch_sample.append(
            #     losses.generalised_energy_distance(stochastic_samples_pred, ema_stochastic_samples_pred,pairwise_dist_metric)[0])
            if region_level:
                consistency_dist_each_batch_sample.append(
                    losses.generalised_energy_distance_region_wise(stochastic_samples_pred, ema_stochastic_samples_pred,
                                                       pairwise_dist_metric, weight_div1, weight_div2,
                                                       num_of_region_h, num_of_region_w, num_of_region_d)[0])
            else:
                consistency_dist_each_batch_sample.append(
                    losses.generalised_energy_distance(stochastic_samples_pred, ema_stochastic_samples_pred,
                                                       pairwise_dist_metric, weight_div1, weight_div2)[0])
    consistency_dist = torch.stack(consistency_dist_each_batch_sample).mean()
    consistency_loss = consistency_weight * consistency_dist
    return consistency_loss, consistency_dist

def get_unsupervised_loss_generalized_dice(consistency_weight, output_logits, ema_output_logits):
    batch_size = output_logits.size()[0]
    output_pred = F.softmax(output_logits, dim=1)
    ema_output_pred = F.softmax(ema_output_logits, dim=1)
    # consistency_dist_each_batch_sample = []
    # for batch_sample_id in range(batch_size):
    #     consistency_dist_each_batch_sample.append(losses.dice_loss(output_pred[batch_sample_id], ema_output_pred[batch_sample_id]))
    # consistency_dist = torch.stack(consistency_dist_each_batch_sample).mean()
    consistency_dist = losses.dice_loss(output_pred, ema_output_pred)
    consistency_loss = consistency_weight * consistency_dist
    return consistency_loss, consistency_dist
def get_unsupervised_loss_self_info(consistency_weight, output_logits, ema_output_logits):
    pred = F.softmax(output_logits, dim=1)  # (b,c,h,w,d)
    pred_entropy_map = losses.entropy_loss_map(pred) # (b,h,w,d)
    ema_pred = F.softmax(ema_output_logits, dim=1)  # (b,c,h,w,d)
    ema_pred_entropy_map = losses.entropy_loss_map(ema_pred)
    consistency_dist = torch.mean((pred_entropy_map - ema_pred_entropy_map) ** 2)
    consistency_loss = consistency_weight * consistency_dist
    return consistency_loss, consistency_dist

def get_unsupervised_loss_energy_based_self_info(consistency_weight, output_logits, ema_output_logits):
    pred = F.softmax(output_logits, dim=1)  # (b,c,h,w,d)
    pred_entropy_map = losses.entropy_loss_map(pred) # (b,h,w,d)
    ema_pred = F.softmax(ema_output_logits, dim=1)  # (b,c,h,w,d)
    ema_pred_entropy_map = losses.entropy_loss_map(ema_pred)
    consistency_dist = torch.mean((pred_entropy_map - ema_pred_entropy_map) ** 2)
    consistency_loss = consistency_weight * consistency_dist
    return consistency_loss, consistency_dist
def calculate_uncertainty(sem_seg_logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.

    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)
def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W, D = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)
    d_step = 1.0 / float(D)

    num_points = min(H * W * D, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W * D), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 3, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices//D%W).to(torch.float) * w_step # x
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices//D//W).to(torch.float) * h_step # y
    point_coords[:, :, 2] = d_step / 2.0 + (point_indices//(H*W)).to(torch.float) * d_step # z
    return point_indices, point_coords

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W, D) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 3) or (N, Hgrid, Wgrid, 3) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:  # [4, 3072, 2]
        add_dim = True
        point_coords = point_coords.unsqueeze(2)  # torch.Size([bs, 2_class, 1, 3])
        if len(input.size()) == 5: # 3D images
            point_coords = point_coords.unsqueeze(3) # torch.Size([bs, 2_class, 1, 1, 3])
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)  # torch.Size([bs, 2_class, 3072, 1, 1])
    if add_dim:
        output = output.squeeze(3)  # torch.Size([bs, 2_class, 3072, 1])
        if len(input.size()) == 5: # 3D images
            output = output.squeeze(3) # torch.Size([bs, 2_class, 3072])

    return output

def get_pixel_samples_near_boundary(logits, labels):

    dilation_dis = args.near_boundary_range
    channels = 1
    h, w, d= labels.size()[-3:]
    stride = 1
    kernel_size = dilation_dis + 1
    padding = int(kernel_size / 2)
    dilation_kernel = torch.ones(kernel_size, kernel_size, kernel_size)
    dilation_kernel = dilation_kernel.repeat(channels, 1, 1, 1, 1)

    dilation_filter = nn.Conv3d(in_channels=channels, out_channels=1,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                groups=channels, bias=False, padding_mode='replicate')

    dilation_filter.weight.data = dilation_kernel
    dilation_filter.weight.requires_grad = False
    dilation_filter.cuda()
    labels_ready = labels.unsqueeze(1).type(torch.cuda.FloatTensor)
    near_boundary_bg = ~torch.eq((dilation_filter(labels_ready) > 0)[:, :, :h, :w, :d], labels_ready)
    labels_ready = (1-labels).unsqueeze(1).type(torch.cuda.FloatTensor)
    near_boundary_fg = ~torch.eq((dilation_filter(labels_ready) > 0)[:, :, :h, :w, :d], labels_ready)
    near_boundary = torch.logical_or(near_boundary_bg, near_boundary_fg)
    # assert (near_boundary != 2).any()
    return near_boundary

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0] # [bs, c, h,w,d]
    num_sampled = int(num_points * oversample_ratio) # 3072
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device) # torch.Size([bs,, 3072, 2])
    point_logits = point_sample(coarse_logits.permute([0,1,4,2,3]), point_coords, align_corners=False) # torch.Size([bs, 54, 3072])
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits) # torch.Size([4, 1, 3072])
    num_uncertain_points = int(importance_sample_ratio * num_points) # 768
    num_random_points = num_points - num_uncertain_points #
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1] # torch.Size([bs,, 768])
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

# def get_pixel_samples_by_uncertainty(logits, labels, ):
#     uncertainty_map = calculate_uncertainty(logits)
#     point_indices, point_coords = get_uncertain_point_coords_on_grid(
#         uncertainty_map, args.random_sampled_num)
#     return point_indices, point_coords
def get_pixel_samples_by_uncertainty(logits, labels, random_sampled_num = args.random_sampled_num, highest = True):
    assert args.oversample_ratio >= 1
    assert args.importance_sample_ratio <= 1 and args.importance_sample_ratio >= 0
    num_boxes = logits.shape[0]  # batch size
    num_sampled = int(random_sampled_num * args.oversample_ratio)  # oversample_num
    point_coords = torch.rand(num_boxes, num_sampled, 3, device=logits.device)  # torch.Size([bs, oversample_num, 3])
    point_logits = point_sample(logits.permute([0,1,4,2,3]), point_coords, align_corners=False)  # torch.Size([bs, 2, oversample_num])
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = calculate_uncertainty(point_logits)  # torch.Size([bs, 1, oversample_num])
    num_uncertain_points = int(args.importance_sample_ratio * random_sampled_num)  # importance_sample_num
    num_random_points = random_sampled_num - num_uncertain_points  #
    if highest:
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]  # torch.Size([bs, 768])
    else:
        idx = torch.topk(-point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]  # torch.Size([bs, 768])
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
    idx += shift[:, None] # index of vectorized uncertainty
    point_coords = point_coords.view(-1, 3)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 3
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 3, device=logits.device),
            ],
            dim=1,
        )
    return point_coords

def get_pixel_samples_from_near_boundary_by_uncertainty(logits, labels, ):
    assert args.oversample_ratio >= 1
    assert args.importance_sample_ratio <= 1 and args.importance_sample_ratio >= 0
    num_boxes = logits.shape[0]  # batch size
    num_sampled = int(args.random_sampled_num * args.oversample_ratio)  # oversample_num
    # point_coords = torch.rand(num_boxes, num_sampled, 3, device=logits.device)  # torch.Size([bs, oversample_num, 3])
    near_boundary_points_binary = get_pixel_samples_near_boundary(None, labels)
    point_coords_all = []
    for each_vol in range(num_boxes): # Each volume has diff #of near boundary pixels so sample seperately
        point_coords_pre = torch.nonzero(near_boundary_points_binary[each_vol].squeeze(0), as_tuple=True) # from binary existence map to position: [H,W,D]-> (3,N)
        # normalize to the range [0,1]; in (x,y,z) order
        point_coords = torch.zeros_like(torch.stack(point_coords_pre, dim=0)).float()
        point_coords[0] = point_coords_pre[1].float() / patch_size[1] # x
        point_coords[1] = point_coords_pre[0].float() / patch_size[0]  # y
        point_coords[2] = point_coords_pre[2].float() / patch_size[2]  # z
        point_coords = point_coords.permute([1, 0]).unsqueeze(0)

        point_logits = point_sample(logits[each_vol: each_vol + 1].permute([0,1,4,2,3]), point_coords, align_corners=False)  # torch.Size([bs, 2, oversample_num])
        point_uncertainties = calculate_uncertainty(point_logits)  # torch.Size([bs, 1, oversample_num])
        # num_points based on actual length
        num_uncertain_points = min(int(args.importance_sample_ratio * args.random_sampled_num) , point_coords.size()[1]) # importance_sample_num
        num_random_points = args.random_sampled_num - num_uncertain_points  #
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]  # torch.Size([bs, 768])
        point_coords = point_coords.view(-1, 3)[idx.view(-1), :].view(1, num_uncertain_points, 3)
        # append random positions
        if num_random_points > 0:
            point_coords = torch.cat([point_coords,torch.rand(1, num_random_points, 3, device=logits.device),],dim=1)

        point_coords_all.append(point_coords)

    point_coords = torch.cat(point_coords_all, 0)
    # if num_random_points > 0:
    #     point_coords = torch.cat(
    #         [
    #             point_coords,
    #             torch.rand(num_boxes, num_random_points, 3, device=logits.device),
    #         ],
    #         dim=1,
    #     )

    return point_coords
resumed_blocks = {'block_nine','block_eight_up', 'block_eight', 'block_seven_up', 'block_seven', 'block_six_up', 'block_six',
                  'block_five_up', 'block_five', 'block_four', 'block_four_dw', 'block_three_dw', 'block_three', 'block_two_dw',
                  'block_one_dw', 'block_two', 'block_one', }
if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # bn_type = args.bn_type
    # with_mask_ssn = args.with_mask_ssn

    def init_parameters(model):
        load_model_path = "../model/" + args.load_model_name + "/"
        save_mode_path = os.path.join(load_model_path, 'iter_' + str(args.load_epoch_num) + '.pth')
        checkpoint = torch.load(save_mode_path, map_location=torch.device('cpu'))
        # remove parameters after conv9 from checkpoint['model']

        # resumed_state_dict = {}
        # for k, v in checkpoint['model'].items():
        #     if k.split('.')[0] in resumed_blocks:
        #         resumed_state_dict[k] = v
        #
        # model_dict = model.state_dict()
        # model_dict.update(resumed_state_dict)
        # model.load_state_dict(model_dict)
        model.load_state_dict(checkpoint['model'])

    def create_model(ema=False):
        # Network definition
        if args.baseline_noout:
            net = StochasticVNetSupCon_noout(input_channels=1, num_classes=num_classes, normalization=bn_type, has_dropout=True, rank=ssn_rank,
                                             head_normalization=args.head_normalization, head_layer_num=args.head_layer_num)
        else:
            net = StochasticVNetSupCon(input_channels=1, num_classes=num_classes, normalization=bn_type, has_dropout=True,
                                 rank=ssn_rank, head_normalization=args.head_normalization, head_layer_num=args.head_layer_num)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    """data loader and sampler for labeled training"""
    transforms_train = transforms.Compose([
        RandomCrop(patch_size, args.transform_fixed),
        ToTensor(),
    ])
    if data_loader_labeled_aug ==  'RandomRotFlip':
        transforms_train.transforms.insert(0, RandomRotFlip(args.transform_fixed))
    elif data_loader_labeled_aug == 'RandomFlip':
        transforms_train.transforms.insert(0, RandomFlip(args.transform_fixed))
    else:
        print('no data aug')
        # transforms_train.transforms.insert(1, NoAug(args.transform_fixed))
    db_train_labeled = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform=transforms_train,
                                image_list_path=args.image_list_path)

    """data loader and sampler for unlabeled training"""
    transforms_train_unlabeled = transforms.Compose([
        RandomCrop(patch_size, args.transform_fixed),
        ToTensor(),
    ])
    if unlabeled_aug_with_gaussian_blur:
        transforms_train_unlabeled.transforms.insert(0, GaussianBlur())
    if unlabeled_aug_with_resize:
        transforms_train_unlabeled.transforms.insert(0, RandomAffine())
    if unlabeled_aug_with_rotationflip:
        assert not unlabeled_aug_with_flip
        transforms_train_unlabeled.transforms.insert(0, RandomRotFlip(args.transform_fixed))
    if unlabeled_aug_with_flip:
        transforms_train_unlabeled.transforms.insert(0, RandomFlip(args.transform_fixed))

    db_train_unlabeled = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform=transforms_train_unlabeled,
                                  image_list_path=args.image_list_path)


    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_num))
    # batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs,args.sampler_fixed)
    labeled_batch_sampler = LabeledBatchSampler(labeled_idxs, labeled_bs,args.sampler_fixed)
    unlabeled_batch_sampler = UnlabeledBatchSampler(unlabeled_idxs, batch_size - labeled_bs,args.sampler_fixed)
    # todo: del
    # i = 0
    # for id1, id2 in zip(labeled_batch_sampler, unlabeled_batch_sampler):
    #     print(id1 + id2)
    #     i += 1
    #     if i == 10:
    #         break
    # todo: del


    db_test = Pancreas(base_dir=train_data_path,
                      split='test',
                      transform=transforms.Compose([
                          CenterCrop(patch_size),
                          ToTensor()
                      ]),)


    def worker_init_fn(worker_id):
        if args.sampler_fixed:
            random.seed(args.seed)
        else:
            random.seed(args.seed+worker_id)
    labeledtrainloader = DataLoader(db_train_labeled, batch_sampler=labeled_batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    unlabeledtrainloader = DataLoader(db_train_unlabeled, batch_sampler=unlabeled_batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    if args.resume:
        init_parameters(model)
        if args.fix_bn_after_resume:
            modules = model.named_children()
            for name, module in modules:
                # if name not in resumed_blocks:
                #     continue
                if not hasattr(module, 'conv'):
                    continue
                for sub_module in module.conv:
                    if isinstance(sub_module, nn.BatchNorm3d):
                        sub_module.eval()



    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    elif args.consistency_type == 'cls_mean_dice':
        consistency_criterion = losses.class_mean_dice_loss_batch_wise
    elif args.consistency_type in {'ged', 'gd'}:
        consistency_criterion = None
    elif args.consistency_type == 'None':
        consistency_criterion = None
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(labeledtrainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(labeledtrainloader)+1
    lr_ = base_lr
    model.train()

    # unlabeled_iter = iter(unlabeledtrainloader)

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, (labeled_sampled_batch, unlabeled_sampled_batch) in enumerate(zip(labeledtrainloader, unlabeledtrainloader)):
            time2 = time.time()
            # unlabeled_sampled_batch = next(unlabeled_iter)

            unlabeled_volume_batch, unlabel_label_batch, unlabeled_mask_batch = unlabeled_sampled_batch['image'], unlabeled_sampled_batch['label'], unlabeled_sampled_batch['mask']

            volume_batch, label_batch, sampling_mask_batch = torch.cat([labeled_sampled_batch['image'], unlabeled_volume_batch]), \
                                                             torch.cat([labeled_sampled_batch['label'], unlabel_label_batch]), \
                                                             torch.cat([labeled_sampled_batch['mask'], unlabeled_mask_batch])
            if not with_mask_ssn:
                sampling_mask_batch = torch.ones_like(sampling_mask_batch)
                unlabeled_mask_batch = torch.ones_like(unlabeled_mask_batch)
            # push to gpu
            unlabeled_volume_batch, unlabel_label_batch, unlabeled_mask_batch = unlabeled_volume_batch.cuda(), unlabel_label_batch.cuda(), unlabeled_mask_batch.cuda()
            volume_batch, label_batch, sampling_mask_batch = volume_batch.cuda(), label_batch.cuda(), sampling_mask_batch.cuda()

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise * perturbation_weight_ema
            if unlabeled_augT_with_gaussian_blur:
                assert not unlabeled_aug_with_gaussian_blur
                for i in range(ema_inputs.size()[0]):
                    ema_inputs[i] = torchio.transforms.RandomBlur()(ema_inputs[i].cpu()).cuda()


            # todo 1: accepts returned features
            outputs, features, states = model(volume_batch, sampling_mask_batch, detach_cov=uncertainty_consistency==0)

            with torch.no_grad():
                ema_output, ema_features, ema_states = ema_model(ema_inputs, unlabeled_mask_batch, noise_weight = perturbation_weight_feature_ema,
                                                   uniform_range=uniform_range, num_feature_perturbated = num_feature_perturbated)
                ema_output_prob = F.softmax(ema_output, dim=1)
                if oracle_checking:
                    teacher_output_batch, teacher_states_batch = ema_model(volume_batch, sampling_mask_batch)
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            unlabeled_mask_batch_r = unlabeled_mask_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2] + list(patch_size)).cuda()
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs, unlabeled_mask_batch_r)[0]
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape((T, stride, 2)+patch_size)
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)

            mc_samples = losses.fixed_re_parametrization_trick(states['distribution'], num_mc_samples)
            mc_samples_supervised = mc_samples[:, :labeled_bs]
            output_prob = F.softmax(outputs[:labeled_bs], dim=1)
            if no_uncertainty_sup:
                supervised_loss, loss_seg_mc_intg, loss_seg_dice = get_supervised_loss_no_uncertainty(
                    outputs[:labeled_bs], label_batch[:labeled_bs])
            else:
                supervised_loss, loss_seg_mc_intg, loss_seg_dice = get_superivsed_loss(outputs[:labeled_bs],
                                                                                       label_batch[:labeled_bs],
                                                                                       mc_samples_supervised, with_dice)

            # todo: supervised contrastive learning
            if args.with_contrastive_loss:
                if args.contrast_pixel_sampling == 'near_boundary':
                    if args.cross_image_contrast:
                        if args.cross_image_sampling:
                            supervised_features = features[:labeled_bs]
                            feature_size = supervised_features.size()[1]
                            is_sampled = get_pixel_samples_near_boundary(outputs[:labeled_bs], label_batch[:labeled_bs])
                            sampled_features_ready = []
                            sampled_labels_ready = []
                            for volume_id in range(labeled_bs):
                                sampled_features = torch.masked_select(supervised_features[volume_id: volume_id+1], is_sampled)
                                sampled_labels = torch.masked_select(label_batch[:labeled_bs][volume_id: volume_id+1], is_sampled[:, 0])
                                sampled_features_pre_sample = sampled_features.view([-1, 1, feature_size])
                                sampled_labels_pre_sample = sampled_labels.view([-1, 1])
                                sample_count = sampled_labels_pre_sample.size()[0]
                                if sample_count > args.random_sampled_num:
                                    perm = torch.randperm(sampled_labels_pre_sample.size()[0])
                                    idx = perm[:args.random_sampled_num]
                                    sampled_features_single = sampled_features_pre_sample[idx]
                                    sampled_labels_single = sampled_labels_pre_sample[idx]
                                else:
                                    sampled_features_single = sampled_features_pre_sample
                                    sampled_labels_single = sampled_labels_pre_sample
                                sampled_features_ready.append(sampled_features_single)
                                sampled_labels_ready.append(sampled_labels_single)
                            sampled_features_ready = torch.cat(sampled_features_ready, dim=0)
                            sampled_labels_ready = torch.cat(sampled_labels_ready, dim=0)
                            # todo: how to take pixels
                            loss_supervised_contrastive = losses.SupConLoss(temperature=args.temp,
                                                                            contrast_mode=args.contrast_mode)(sampled_features_ready, sampled_labels_ready)
                            running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                            supervised_loss += running_sup_cont_weight * loss_supervised_contrastive
                        else:
                            supervised_features = features[:labeled_bs]
                            feature_size = supervised_features.size()[1]
                            is_sampled = get_pixel_samples_near_boundary(outputs[:labeled_bs], label_batch[:labeled_bs])
                            sampled_features = torch.masked_select(supervised_features, is_sampled)
                            sampled_labels = torch.masked_select(label_batch[:labeled_bs], is_sampled[:,0])
                            sampled_features_pre_sample = sampled_features.view([-1, 1, feature_size])
                            sampled_labels_pre_sample = sampled_labels.view([-1, 1])
                            sample_count = sampled_labels_pre_sample.size()[0]
                            if sample_count > args.random_sampled_num:
                                perm = torch.randperm(sampled_labels_pre_sample.size()[0])
                                idx = perm[:args.random_sampled_num]
                                sampled_features_ready = sampled_features_pre_sample[idx]
                                sampled_labels_ready = sampled_labels_pre_sample[idx]
                            else:
                                sampled_features_ready = sampled_features_pre_sample
                                sampled_labels_ready = sampled_labels_pre_sample

                            # todo: how to take pixels
                            loss_supervised_contrastive = losses.SupConLoss(temperature=args.temp, contrast_mode=args.contrast_mode)(sampled_features_ready,sampled_labels_ready )
                            running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                            supervised_loss +=  running_sup_cont_weight* loss_supervised_contrastive
                    else:
                        if args.cross_image_sampling:
                            supervised_features = features[:labeled_bs]
                            feature_size = supervised_features.size()[1]
                            is_sampled = get_pixel_samples_near_boundary(outputs[:labeled_bs], label_batch[:labeled_bs]) # bs, 1, h, w, d
                            loss_supervised_contrastive = 0
                            for volume_id in range(labeled_bs):
                                sampled_features = torch.masked_select(supervised_features[volume_id: volume_id + 1], is_sampled) #!!! correction after exp1k_002
                                sampled_labels = torch.masked_select(label_batch[:labeled_bs][volume_id: volume_id + 1], is_sampled[:, 0])
                                sampled_features_pre_sample = sampled_features.view([-1, 1, feature_size])
                                sampled_labels_pre_sample = sampled_labels.view([-1, 1])
                                sample_count = sampled_labels_pre_sample.size()[0]
                                if sample_count > args.random_sampled_num:
                                    perm = torch.randperm(sampled_labels_pre_sample.size()[0])
                                    idx = perm[:args.random_sampled_num]
                                    sampled_features_ready = sampled_features_pre_sample[idx]
                                    sampled_labels_ready = sampled_labels_pre_sample[idx]
                                else:
                                    sampled_features_ready = sampled_features_pre_sample
                                    sampled_labels_ready = sampled_labels_pre_sample
                                loss_supervised_contrastive += losses.SupConLoss(temperature=args.temp, contrast_mode=args.contrast_mode)(sampled_features_ready, sampled_labels_ready)
                            running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                            supervised_loss += running_sup_cont_weight * loss_supervised_contrastive / labeled_bs
                        else:
                            supervised_features = features[:labeled_bs]
                            feature_size = supervised_features.size()[1]
                            is_sampled = get_pixel_samples_near_boundary(outputs[:labeled_bs],
                                                                         label_batch[:labeled_bs])  # bs, 1, h, w, d
                            loss_supervised_contrastive = 0
                            v_num = 0
                            for volume_id in range(labeled_bs):
                                # todo add to la

                                if len(torch.nonzero(is_sampled[volume_id: volume_id + 1]))==0:
                                    continue
                                v_num += 1
                                sampled_features = torch.masked_select(supervised_features[volume_id: volume_id + 1],
                                                                       is_sampled[volume_id: volume_id + 1])  # !!! correction after exp1k_002
                                sampled_labels = torch.masked_select(label_batch[:labeled_bs][volume_id: volume_id + 1],is_sampled[volume_id: volume_id + 1, 0])
                                sampled_features_pre_sample = sampled_features.view([-1, 1, feature_size])
                                sampled_labels_pre_sample = sampled_labels.view([-1, 1])
                                sample_count = sampled_labels_pre_sample.size()[0]
                                if sample_count > args.random_sampled_num:
                                    perm = torch.randperm(sampled_labels_pre_sample.size()[0])
                                    idx = perm[:args.random_sampled_num]
                                    sampled_features_ready = sampled_features_pre_sample[idx]
                                    sampled_labels_ready = sampled_labels_pre_sample[idx]
                                else:
                                    sampled_features_ready = sampled_features_pre_sample
                                    sampled_labels_ready = sampled_labels_pre_sample
                                loss_supervised_contrastive += losses.SupConLoss(temperature=args.temp,
                                                                                 contrast_mode=args.contrast_mode)(
                                    sampled_features_ready, sampled_labels_ready)
                            running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                            supervised_loss += running_sup_cont_weight * loss_supervised_contrastive / v_num

                elif args.contrast_pixel_sampling == 'uncertainty':
                    if args.cross_image_contrast:
                        supervised_features = features[:labeled_bs]
                        feature_size = supervised_features.size()[1]
                        point_coords = get_pixel_samples_by_uncertainty(outputs[:labeled_bs], label_batch[:labeled_bs])
                        sampled_features_pre = point_sample(supervised_features, point_coords, align_corners=False)
                        sampled_labels_pre = point_sample(label_batch[:labeled_bs].float().unsqueeze(1), point_coords,  align_corners=False).squeeze(1).long()
                        sampled_features_ready = sampled_features_pre.permute([0,2,1]).contiguous().view([-1, 1, feature_size])
                        sampled_labels_ready = sampled_labels_pre.view([-1, 1])
                        loss_supervised_contrastive = losses.SupConLoss(temperature=args.temp,contrast_mode=args.contrast_mode)(sampled_features_ready, sampled_labels_ready)
                        running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                        supervised_loss += running_sup_cont_weight * loss_supervised_contrastive
                    else:
                        if args.cross_image_sampling:
                            supervised_features = features[:labeled_bs]
                            feature_size = supervised_features.size()[1]
                            point_coords = get_pixel_samples_by_uncertainty(outputs[:labeled_bs],label_batch[:labeled_bs])
                            point_coords_all = torch.cat(torch.chunk(point_coords, chunks=labeled_bs, dim=0),dim=1)  # bs,N_points, 3 -> 1, bs*N_points, 3
                            loss_supervised_contrastive = 0
                            for volume_id in range(labeled_bs):
                                sampled_features_pre = point_sample(supervised_features[volume_id: volume_id + 1],
                                                                    point_coords_all, align_corners=False)
                                sampled_labels_pre = point_sample(label_batch[:labeled_bs][volume_id: volume_id + 1].float().unsqueeze(1),
                                                                  point_coords_all,align_corners=False).squeeze(1).long()
                                sampled_features_ready = sampled_features_pre.permute([0, 2, 1]).contiguous().view([-1, 1, feature_size])
                                sampled_labels_ready = sampled_labels_pre.view([-1, 1])
                                loss_supervised_contrastive += losses.SupConLoss(temperature=args.temp, contrast_mode=args.contrast_mode)(sampled_features_ready, sampled_labels_ready)
                            loss_supervised_contrastive /= labeled_bs
                            running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                            supervised_loss += running_sup_cont_weight * loss_supervised_contrastive

                elif args.contrast_pixel_sampling == 'near_boundary_uncertainty':
                    if args.cross_image_contrast:
                        supervised_features = features[:labeled_bs]
                        feature_size = supervised_features.size()[1]
                        point_coords = get_pixel_samples_from_near_boundary_by_uncertainty(outputs[:labeled_bs], label_batch[:labeled_bs])
                        sampled_features_pre = point_sample(supervised_features, point_coords, align_corners=False)
                        sampled_labels_pre = point_sample(label_batch[:labeled_bs].float().unsqueeze(1), point_coords,align_corners=False).squeeze(1).long()
                        sampled_features_ready = sampled_features_pre.permute([0, 2, 1]).contiguous().view(
                            [-1, 1, feature_size])
                        sampled_labels_ready = sampled_labels_pre.view([-1, 1])
                        loss_supervised_contrastive = losses.SupConLoss(temperature=args.temp,contrast_mode=args.contrast_mode)(sampled_features_ready, sampled_labels_ready)
                        running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                        supervised_loss += running_sup_cont_weight * loss_supervised_contrastive
                    else:
                        if args.cross_image_sampling:
                            loss_supervised_contrastive = 0
                            supervised_features = features[:labeled_bs]
                            feature_size = supervised_features.size()[1]
                            point_coords_all = get_pixel_samples_from_near_boundary_by_uncertainty(outputs[:labeled_bs],label_batch[:labeled_bs])
                            point_coords_all = torch.cat(torch.chunk(point_coords_all, chunks=labeled_bs, dim=0), dim = 1) # bs,N_points, 3 -> 1, bs*N_points, 3
                            for volume_id in range(labeled_bs):
                                sampled_features_pre = point_sample(supervised_features[volume_id: volume_id + 1],
                                                                    point_coords_all, align_corners=False)
                                sampled_labels_pre = point_sample(
                                    label_batch[:labeled_bs][volume_id: volume_id + 1].float().unsqueeze(1),
                                    point_coords_all,
                                    align_corners=False).squeeze(1).long()
                                sampled_features_ready = sampled_features_pre.permute([0, 2, 1]).contiguous().view(
                                    [-1, 1, feature_size])
                                sampled_labels_ready = sampled_labels_pre.view([-1, 1])
                                loss_supervised_contrastive += losses.SupConLoss(temperature=args.temp,
                                                                                 contrast_mode=args.contrast_mode)(sampled_features_ready, sampled_labels_ready)
                            loss_supervised_contrastive /= labeled_bs
                            running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                            supervised_loss += running_sup_cont_weight * loss_supervised_contrastive
                        else:
                            loss_supervised_contrastive = 0
                            supervised_features = features[:labeled_bs]
                            feature_size = supervised_features.size()[1]
                            point_coords_all = []
                            for volume_id in range(labeled_bs):
                                point_coords_single = get_pixel_samples_from_near_boundary_by_uncertainty(outputs[:labeled_bs][volume_id : volume_id + 1],
                                                                                                   label_batch[:labeled_bs][volume_id : volume_id + 1])
                                sampled_features_pre = point_sample(supervised_features[volume_id : volume_id + 1], point_coords_single, align_corners=False)
                                sampled_labels_pre = point_sample(label_batch[:labeled_bs][volume_id : volume_id + 1].float().unsqueeze(1), point_coords_single,
                                                                  align_corners=False).squeeze(1).long()
                                sampled_features_ready = sampled_features_pre.permute([0, 2, 1]).contiguous().view(
                                    [-1, 1, feature_size])
                                sampled_labels_ready = sampled_labels_pre.view([-1, 1])
                                loss_supervised_contrastive += losses.SupConLoss(temperature=args.temp,
                                                                                contrast_mode=args.contrast_mode)(sampled_features_ready, sampled_labels_ready)
                                point_coords_all.append(point_coords_single)
                            point_coords = torch.cat(point_coords_all, dim=0)

                            loss_supervised_contrastive /= labeled_bs
                            running_sup_cont_weight = get_sup_cont_weight(iter_num // 150, args.sup_cont_weight)
                            supervised_loss += running_sup_cont_weight * loss_supervised_contrastive


            else:
                loss_supervised_contrastive = torch.zeros([1])
                running_sup_cont_weight = 0


            if args.with_unsup_contrastive_loss:
                if args.cross_unsup_image_contrast:
                    unsupervised_features = features[labeled_bs:]
                    feature_size = unsupervised_features.size()[1]
                    unsup_pseudo_label = torch.argmax(outputs[labeled_bs:], dim=1)
                    unsup_point_coords = get_pixel_samples_by_uncertainty(outputs[labeled_bs:], unsup_pseudo_label,
                                                                    random_sampled_num=args.random_unsup_sampled_num,
                                                                    highest=False)
                    sampled_unsup_features_pre = point_sample(unsupervised_features, unsup_point_coords, align_corners=False)
                    sampled_unsup_pseudo_labels_pre = point_sample(unsup_pseudo_label.float().unsqueeze(1), unsup_point_coords,
                                                      align_corners=False).squeeze(1).long()
                    sampled_unsup_features_ready = sampled_unsup_features_pre.permute([0, 2, 1]).contiguous().view(
                        [-1, 1, feature_size])
                    sampled_unsup_pseudo_labels_ready = sampled_unsup_pseudo_labels_pre.view([-1, 1])
                    loss_unsupervised_contrastive = losses.SupConLoss(temperature=args.temp,
                                                                    contrast_mode=args.contrast_mode)(
                        sampled_unsup_features_ready, sampled_unsup_pseudo_labels_ready)
                    running_unsup_cont_weight = get_unsup_cont_weight(iter_num // 150, args.unsup_cont_weight)
                    supervised_loss += running_unsup_cont_weight * loss_unsupervised_contrastive
                else:
                    if args.cross_unsup_image_sampling:
                        assert False
                    else:
                        unsupervised_features = features[labeled_bs:]
                        feature_size = unsupervised_features.size()[1]
                        unsup_pseudo_label = torch.argmax(outputs[labeled_bs:], dim=1)
                        unsup_point_coords = get_pixel_samples_by_uncertainty(outputs[labeled_bs:], unsup_pseudo_label,
                                                                              random_sampled_num=args.random_unsup_sampled_num,
                                                                              highest=False)

                        loss_unsupervised_contrastive = 0
                        for volume_id in range(batch_size - labeled_bs):
                            sampled_unsup_features_pre = point_sample(unsupervised_features[volume_id: volume_id + 1],
                                                                unsup_point_coords[volume_id: volume_id + 1], align_corners=False)
                            sampled_pseudo_labels_pre = point_sample(
                                unsup_pseudo_label[volume_id: volume_id + 1].float().unsqueeze(1),
                                unsup_point_coords[volume_id: volume_id + 1], align_corners=False).squeeze(1).long()
                            sampled_unsup_features_ready = sampled_unsup_features_pre.permute([0, 2, 1]).contiguous().view(
                                [-1, 1, feature_size])
                            sampled_pseudo_labels_ready = sampled_pseudo_labels_pre.view([-1, 1])
                            loss_unsupervised_contrastive += losses.SupConLoss(temperature=args.temp,
                                                                             contrast_mode=args.contrast_mode)(
                                sampled_unsup_features_ready, sampled_pseudo_labels_ready)
                        loss_unsupervised_contrastive /= labeled_bs
                        running_unsup_cont_weight = get_unsup_cont_weight(iter_num // 150, args.unsup_cont_weight)
                        supervised_loss += running_unsup_cont_weight * loss_unsupervised_contrastive

            # import pdb
            #
            # pdb.set_trace()
            if oracle_checking:
                with torch.no_grad(): # oracle testing
                    supervised_loss_student_batch, loss_student_mc_intg, loss_student_dice, dice_coeff_student = oracle_check(outputs,label_batch,mc_samples)
                    mc_samples_teacher_batch = losses.fixed_re_parametrization_trick(teacher_states_batch['distribution'], num_mc_samples)
                    supervised_loss_teacher_batch, loss_teacher_mc_intg, loss_teacher_dice, dice_coeff_teacher = oracle_check(teacher_output_batch, label_batch, mc_samples_teacher_batch)


            # if args.consistency == 0:
            #     consistency_weight = torch.zeros([1])
            #     consistency_loss, consistency_dist = torch.zeros([1]),torch.zeros([1])
            #     loss = supervised_loss
            # else:
            #     consistency_weight = get_current_consistency_weight(iter_num // 150)
            #     mc_samples_unsupervised = mc_samples[:, labeled_bs:]
            #     mc_samples_ema = losses.fixed_re_parametrization_trick(ema_states['distribution'], num_mc_samples)
            #     consistency_loss, consistency_dist = get_unsupervised_loss(consistency_weight, mc_samples_unsupervised,
            #                                                                mc_samples_ema)
            #     loss = supervised_loss + consistency_loss
            if with_uncertainty_mask:
                threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                uncertainty_mask = (uncertainty < threshold).float()
            else:
                uncertainty_mask = torch.ones_like(uncertainty)
            uncertainty_mask_tile = uncertainty_mask.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)


            consistency_weight = get_current_consistency_weight(iter_num // 150, args.consistency)
            mc_samples_unsupervised = mc_samples[:, labeled_bs:]
            mc_samples_ema = losses.fixed_re_parametrization_trick(ema_states['distribution'], num_mc_samples)
            if args.consistency_type == 'ged':
                consistency_loss, consistency_dist = get_unsupervised_loss(consistency_weight, mc_samples_unsupervised * uncertainty_mask_tile,
                                                                           mc_samples_ema * uncertainty_mask_tile)
                if uncertainty_consistency != 0:
                    consistency_weight_uncertainty = get_current_consistency_weight(iter_num // 150, args.uncertainty_consistency)

                    if cov_diag_consistency_weight != 0:
                        cov_diag_consistency_dist = torch.mean((states['cov_diag'][labeled_bs:] - ema_states['cov_diag']) ** 2)
                        cov_diag_consistency_loss = consistency_weight_uncertainty * cov_diag_consistency_dist * cov_diag_consistency_weight

                    else:
                        cov_diag_consistency_dist, cov_diag_consistency_loss = torch.tensor(0), torch.tensor(0)
                    if cov_factor_consistency_weight != 0:
                        cov_factor_consistency_dist = torch.mean((states['cov_factor'][labeled_bs:] - ema_states['cov_factor']) ** 2)
                        cov_factor_consistency_loss = consistency_weight_uncertainty * cov_factor_consistency_dist * cov_factor_consistency_weight
                    else:
                        cov_factor_consistency_dist, cov_factor_consistency_loss = torch.tensor(0), torch.tensor(0)
                    consistency_loss = consistency_loss +  cov_diag_consistency_loss + cov_factor_consistency_loss
            elif args.consistency_type == 'gd': # generalized dice
                consistency_loss, consistency_dist = get_unsupervised_loss_generalized_dice(consistency_weight,
                                                                           outputs[labeled_bs:] * uncertainty_mask,
                                                                           ema_output * uncertainty_mask)
                if uncertainty_consistency != 0:
                    consistency_weight_uncertainty = get_current_consistency_weight(iter_num // 150, args.uncertainty_consistency)
                    if cov_diag_consistency_weight != 0:
                        cov_diag_consistency_dist = torch.mean(
                            (states['cov_diag'][labeled_bs:] - ema_states['cov_diag']) ** 2)
                        cov_diag_consistency_loss = consistency_weight_uncertainty * cov_diag_consistency_dist * cov_diag_consistency_weight

                    else:
                        cov_diag_consistency_dist, cov_diag_consistency_loss = torch.tensor(0), torch.tensor(0)
                    if cov_factor_consistency_weight != 0:
                        cov_factor_consistency_dist = torch.mean(
                            (states['cov_factor'][labeled_bs:] - ema_states['cov_factor']) ** 2)
                        cov_factor_consistency_loss = consistency_weight_uncertainty * cov_factor_consistency_dist * cov_factor_consistency_weight
                    else:
                        cov_factor_consistency_dist, cov_factor_consistency_loss = torch.tensor(0), torch.tensor(0)
                    consistency_loss = consistency_loss + cov_diag_consistency_loss + cov_factor_consistency_loss
            elif args.consistency_type == 'None':
                consistency_loss, consistency_dist = torch.tensor(0), torch.tensor(0)

            else: # UA-MT with stochastic supervision
                # assert with_uncertainty_mask
                consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output)  # (batch, 2, 112,112,80)
                consistency_dist = torch.sum(uncertainty_mask * consistency_dist) / (2 * torch.sum(uncertainty_mask) + 1e-16)
                consistency_loss = consistency_weight * consistency_dist
                if uncertainty_consistency != 0:
                    consistency_weight_uncertainty = get_current_consistency_weight(iter_num // 150, args.uncertainty_consistency)
                    if cov_diag_consistency_weight != 0:
                        cov_diag_consistency_dist = torch.mean(
                            (states['cov_diag'][labeled_bs:] - ema_states['cov_diag']) ** 2)
                        cov_diag_consistency_loss = consistency_weight_uncertainty * cov_diag_consistency_dist * cov_diag_consistency_weight

                    else:
                        cov_diag_consistency_dist, cov_diag_consistency_loss = torch.tensor(0), torch.tensor(0)
                    if cov_factor_consistency_weight != 0:
                        cov_factor_consistency_dist = torch.mean(
                            (states['cov_factor'][labeled_bs:] - ema_states['cov_factor']) ** 2)
                        cov_factor_consistency_loss = consistency_weight_uncertainty * cov_factor_consistency_dist * cov_factor_consistency_weight
                    else:
                        cov_factor_consistency_dist, cov_factor_consistency_loss = torch.tensor(0), torch.tensor(0)
                    consistency_loss = consistency_loss + cov_diag_consistency_loss + cov_factor_consistency_loss

            if lambda_g != 0:
                consistency_weight_self_info = get_current_consistency_weight(iter_num // 150, lambda_g)
                if energy_g:
                    consistency_self_info_loss, consistency_self_info_dist = get_unsupervised_loss_energy_based_self_info(
                        consistency_weight_self_info,outputs[labeled_bs:] * uncertainty_mask,ema_output * uncertainty_mask)
                else:
                    consistency_self_info_loss, consistency_self_info_dist = get_unsupervised_loss_self_info(
                        consistency_weight_self_info, outputs[labeled_bs:], ema_output)
            else:
                consistency_self_info_loss = 0

            if args.with_unsup_contrastive_subvolume_after_decoder:
                feature_downsampled = torch.nn.AvgPool3d(args.unsup_average_size)(features[labeled_bs:])  # bs, fv_l, 96/4, 96/4, 94/4
                feature_downsampled_ema = torch.nn.AvgPool3d(args.unsup_average_size)(ema_features)  # bs, fv_l, 96/4, 96/4, 94/4
                feature_size = feature_downsampled.size()[1]
                feature_downsampled_ready = feature_downsampled.view([batch_size - labeled_bs, feature_size, -1]).permute([0, 2, 1]).contiguous().view([-1, 1, feature_size])
                feature_downsampled_ema_ready = feature_downsampled_ema.view([batch_size - labeled_bs, feature_size, -1]).permute([0, 2, 1]).contiguous().view([-1, 1, feature_size])
                if args.subvolume_after_decoder_cross_image_contrast:
                    feature_downsampled_after_decoder_ready = torch.cat([feature_downsampled_ready, feature_downsampled_ema_ready], dim=0) # bs_unlabel * pixel_num * 2, feature_size
                    spatial_label_single = torch.arange(0, (batch_size - labeled_bs) * np.prod(patch_size) / np.power(args.unsup_average_size, 3)).view([-1]).long()
                    spatial_labels_ready = torch.cat([spatial_label_single, spatial_label_single], dim=0) # batch_size - labeled_bs, patch_size[0] / args.unsup_average_size, patch_size[1] / args.unsup_average_size, patch_size[2] / args.unsup_average_size,
                    loss_unsup_contrastive_subvolume_after_dec = losses.SupConLoss(temperature=args.temp, contrast_mode=args.contrast_mode)\
                        (feature_downsampled_after_decoder_ready, spatial_labels_ready)

                else:
                    spatial_label_single = torch.arange(0, np.prod(patch_size) / np.power(args.unsup_average_size, 3)).view([-1]).long()
                    spatial_labels_ready = torch.cat([spatial_label_single, spatial_label_single],
                                                     dim=0)
                    loss_unsup_contrastive_subvolume_after_dec = torch.zeros([1])
                    for volume_id in range(batch_size - labeled_bs):
                        feature_downsampled_after_decoder_ready = torch.cat(
                            [feature_downsampled_ready[volume_id: volume_id + 1], feature_downsampled_ema_ready[volume_id: volume_id + 1]], dim=0)
                        loss_unsup_contrastive_subvolume_after_dec += losses.SupConLoss(temperature=args.temp, contrast_mode=args.contrast_mode)\
                        (feature_downsampled_after_decoder_ready, spatial_labels_ready)
                    loss_unsup_contrastive_subvolume_after_dec /= (batch_size-labeled_bs)



            loss = supervised_loss + consistency_loss + consistency_self_info_loss
            if args.with_unsup_contrastive_subvolume_after_decoder:
                running_unsup_cont_subvolume_after_decoder_weight = get_unsup_cont_weight(iter_num // 150, args.unsup_subvolume_after_decoder_cont_weight,
                                                                                          scheme=args.unsup_subvolume_after_decoder_cont_rampup_scheme, ramp_up=args.unsup_subvolume_after_decoder_rampup)
                loss += loss_unsup_contrastive_subvolume_after_dec * running_unsup_cont_subvolume_after_decoder_weight
            # loss = supervised_loss + consistency_loss





            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.ema_dec_only:
                update_ema_variables_decoder_only(model, ema_model, args.ema_decay, iter_num)
            elif args.ema_enc_only:
                update_ema_variables_encoder_only(model, ema_model, args.ema_decay, iter_num)
            else:
                update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            # writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            # writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg_mc_intg', loss_seg_mc_intg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            # todo: log new loss
            writer.add_scalar('loss/loss_supervised_contrastive', loss_supervised_contrastive, iter_num)
            writer.add_scalar('loss/running_sup_cont_weight', running_sup_cont_weight, iter_num)
            if args.with_unsup_contrastive_loss:
                writer.add_scalar('loss/loss_unsupervised_contrastive', loss_unsupervised_contrastive, iter_num)
                writer.add_scalar('loss/running_unsup_cont_weight', running_unsup_cont_weight, iter_num)
            if args.subvolume_after_decoder_cross_image_contrast:
                writer.add_scalar('loss/loss_unsup_contrastive_subvolume_after_dec', loss_unsup_contrastive_subvolume_after_dec, iter_num)
                writer.add_scalar('loss/running_unsup_cont_subvolume_after_decoder_weight',running_unsup_cont_subvolume_after_decoder_weight, iter_num)


            if ab_ce:
                writer.add_scalar('loss/anneal_threshold', anneal_threshold, iter_num)

            if lambda_g != 0:
                writer.add_scalar('train/consistency_self_info_loss', consistency_self_info_loss, iter_num)
                writer.add_scalar('train/consistency_weight_self_info', consistency_weight_self_info, iter_num)
                writer.add_scalar('train/consistency_self_info_dist', consistency_self_info_dist, iter_num)

            if oracle_checking:
                writer.add_scalars('loss_batch_ST/loss_ST', {'student': supervised_loss_student_batch, 'teacher': supervised_loss_teacher_batch}, iter_num)
                writer.add_scalars('loss_batch_ST/loss_seg_mc_intg_ST', {'student': loss_student_mc_intg, 'teacher': loss_teacher_mc_intg}, iter_num)
                writer.add_scalars('loss_batch_ST/loss_seg_dice_ST', {'student': loss_student_dice, 'teacher': loss_teacher_dice}, iter_num)
                writer.add_scalars('loss_batch_ST/error_rate_dice_ST', {'student': dice_coeff_student, 'teacher': dice_coeff_teacher}, iter_num)

                writer.add_scalar('loss_batch_S_minus_T/loss_S_minus_T', supervised_loss_student_batch - supervised_loss_teacher_batch, iter_num)
                writer.add_scalar('loss_batch_S_minus_T/loss_seg_mc_intg_S_minus_T', loss_student_mc_intg - loss_teacher_mc_intg, iter_num)
                writer.add_scalar('loss_batch_S_minus_T/loss_seg_dice_S_minus_T', loss_student_dice - loss_teacher_dice, iter_num)
                writer.add_scalar('loss_batch_ST/error_rate_dice_S_minus_T',dice_coeff_student - dice_coeff_teacher, iter_num)

            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)


            if uncertainty_consistency != 0:
                writer.add_scalar('train/consistency_weight_uncertainty', consistency_weight_uncertainty, iter_num)
                writer.add_scalar('train/cov_diag_consistency_dist', cov_diag_consistency_dist, iter_num)
                writer.add_scalar('train/cov_factor_consistency_dist', cov_factor_consistency_dist, iter_num)
                writer.add_scalar('train/cov_diag_consistency_loss', cov_factor_consistency_loss, iter_num)
                writer.add_scalar('train/cov_factor_consistency_loss', cov_factor_consistency_loss, iter_num)

            if uncertainty_consistency == 0:
                if ab_ce:
                    logging.info('iteration %d : loss : %f anneal_threshold: %f cons_dist: %f, loss_weight: %f' %
                                 (iter_num, loss.item(), anneal_threshold, consistency_dist.item(),
                                  consistency_weight))
                else:
                    if args.with_unsup_contrastive_loss:
                        if args.with_unsup_contrastive_subvolume_after_decoder:
                            basic_info = 'iteration %d : loss : %f supervised_loss: %f, loss_seg_mc_intg: %f, loss_seg_dice: %f,  loss_supervised_contrastive: %f, running_sup_cont_weight: %f,' \
                                         ' loss_unsupervised_contrastive: %f, running_unsup_cont_weight: %f ' \
                                         'loss_unsup_contrastive_subvolume_after_dec: %f, running_unsup_cont_subvolume_after_decoder_weight: %f,'\
                                         ' cons_dist: %f, loss_weight: %f ' % \
                                         (iter_num, loss.item(), supervised_loss.item(), loss_seg_mc_intg.item(),
                                          loss_seg_dice.item(), loss_supervised_contrastive.item(),
                                          running_sup_cont_weight,
                                          loss_unsupervised_contrastive.item(), running_unsup_cont_weight,
                                          loss_unsup_contrastive_subvolume_after_dec.item(),
                                          running_unsup_cont_subvolume_after_decoder_weight,
                                          consistency_dist.item(),
                                          consistency_weight)
                        else:
                            basic_info = 'iteration %d : loss : %f supervised_loss: %f, loss_seg_mc_intg: %f, loss_seg_dice: %f,  loss_supervised_contrastive: %f, running_sup_cont_weight: %f,' \
                                     ' loss_unsupervised_contrastive: %f, running_unsup_cont_weight: %f '\
                                     ' cons_dist: %f, loss_weight: %f ' % \
                                     (iter_num, loss.item(), supervised_loss.item(), loss_seg_mc_intg.item(),
                                      loss_seg_dice.item(), loss_supervised_contrastive.item(), running_sup_cont_weight,
                                      loss_unsupervised_contrastive.item(), running_unsup_cont_weight,
                                      consistency_dist.item(),
                                      consistency_weight)
                    else:
                        if args.with_unsup_contrastive_subvolume_after_decoder:
                            basic_info = 'iteration %d : loss : %f supervised_loss: %f, loss_seg_mc_intg: %f, loss_seg_dice: %f,  loss_supervised_contrastive: %f, running_sup_cont_weight: %f,' \
                                         'loss_unsup_contrastive_subvolume_after_dec: %f, running_unsup_cont_subvolume_after_decoder_weight: %f, cons_dist: %f, loss_weight: %f ' % \
                                         (iter_num, loss.item(), supervised_loss.item(), loss_seg_mc_intg.item(),
                                          loss_seg_dice.item(), loss_supervised_contrastive.item(),
                                          running_sup_cont_weight, loss_unsup_contrastive_subvolume_after_dec.item(),
                                          running_unsup_cont_subvolume_after_decoder_weight,
                                          consistency_dist.item(),
                                          consistency_weight)
                        else:
                            basic_info = 'iteration %d : loss : %f supervised_loss: %f, loss_seg_mc_intg: %f, loss_seg_dice: %f,  loss_supervised_contrastive: %f, running_sup_cont_weight: %f, cons_dist: %f, loss_weight: %f ' % \
                                 (iter_num, loss.item(), supervised_loss.item(), loss_seg_mc_intg.item(), loss_seg_dice.item(),loss_supervised_contrastive.item(), running_sup_cont_weight, consistency_dist.item(),
                                  consistency_weight)
                    if lambda_g != 0:
                        self_info_related_info = 'consistency_self_info_loss: %f consistency_weight_self_info: %f consistency_self_info_dist: %f' % \
                                             (consistency_self_info_loss.item(), consistency_weight_self_info, consistency_self_info_dist.item())
                    else:
                        self_info_related_info = ''
                    logging.info(basic_info +  self_info_related_info)

                    # logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                    #      (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            else:
                logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f, '
                             'cov_diag_consistency_dist: %f, cov_factor_consistency_dist: %f, consistency_weight_uncertainty :%f' %
                             (iter_num, loss.item(), consistency_dist.item(), consistency_weight, cov_diag_consistency_dist.item(), cov_factor_consistency_dist.item(), consistency_weight_uncertainty))
            if iter_num % 50 == 0:
                # monitor the performance of each branch
                labeled_output_prob = output_prob
                unlabeled_output_prob = F.softmax(outputs[labeled_bs:batch_size], dim=1)

                labeled_output_pred = torch.argmax(labeled_output_prob, dim=1)
                unlabeled_output_pred = torch.argmax(unlabeled_output_prob, dim=1)
                ema_output_pred = torch.argmax(ema_output_prob, dim=1)

                metric_labeled = test_batch(labeled_output_pred.cpu().data.numpy(),
                                            label_batch[:labeled_bs].cpu().data.numpy(), num_classes=num_classes)
                metric_unlabeled = test_batch(unlabeled_output_pred.cpu().data.numpy(),
                                              label_batch[labeled_bs:batch_size].cpu().data.numpy(),
                                              num_classes=num_classes)
                metric_unlabeled_ema = test_batch(ema_output_pred.cpu().data.numpy(),
                                                  label_batch[labeled_bs:batch_size].cpu().data.numpy(),
                                                  num_classes=num_classes)
                for i in range(num_classes - 1):
                    writer.add_scalars('train_evaluator/dice_class{}'.format(i+1),
                                       {'labeled': metric_labeled[i][0], 'unlabeled': metric_unlabeled[i][0],
                                        'unlabeled_ema': metric_unlabeled_ema[i][0]}, iter_num)
                    writer.add_scalars('train_evaluator/hd95_class{}'.format(i+1),
                                       {'labeled': metric_labeled[i][1], 'unlabeled': metric_unlabeled[i][1],
                                        'unlabeled_ema': metric_unlabeled_ema[i][1]}, iter_num)

                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(output_prob[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)


                if args.with_contrastive_loss:
                    if args.contrast_pixel_sampling == 'near_boundary':
                        image = is_sampled.type(torch.cuda.IntTensor)[0, 0, :, :, 20:61:10].permute(2, 0, 1)
                        grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                        writer.add_image('train/NearBoundary', grid_image, iter_num)

                    elif args.contrast_pixel_sampling in ('uncertainty', 'near_boundary_uncertainty') :
                        # todo 1: subplot
                        # todo 2: confidence by logits; confidence by pred
                        # todo 3: diff z instead of 20
                        fig = plt.figure(figsize=(10, 10))

                        # plt.xticks([], [])
                        # plt.yticks([], [])
                        # plt.imshow(mask, interpolation="nearest", cmap=plt.get_cmap('gray'))
                        if point_coords is not None:
                            ax1 = plt.subplot(151)
                            plt.imshow(volume_batch[0, 0, :, :, 40].cpu(), cmap=plt.get_cmap('gray'))
                            to_plot = torch.masked_select(point_coords[0], torch.logical_and(point_coords[0,:,-1]*patch_size[-1]>=40, point_coords[0,:,-1]*patch_size[-1]<41).unsqueeze(-1)).view([-1,3])
                            plt.scatter(x=to_plot[:,0].cpu()*patch_size[0], y=to_plot[:,1].cpu()*patch_size[1], color="red", s=2, alpha=0.6)
                            ax2 = plt.subplot(152)
                            uncertainty_by_pred = (torch.min(output_prob[0, :, :, :, 40], dim=0)[0] - torch.max(output_prob[0, :, :, :, 40], dim=0)[0]).detach().cpu().numpy()
                            plt.imshow(uncertainty_by_pred, cmap=plt.get_cmap('gray'))
                            plt.scatter(x=to_plot[:, 0].cpu() * patch_size[0], y=to_plot[:, 1].cpu() * patch_size[1],
                                        color="red", s=2, alpha=0.6)
                            ax3 = plt.subplot(153)
                            uncertainty_by_logits = (torch.min(outputs[0, :, :, :, 40], dim=0)[0] - torch.max(outputs[0, :, :, :, 40], dim=0)[0]).detach().cpu().numpy()
                            plt.imshow(uncertainty_by_logits, cmap=plt.get_cmap('gray'))
                            plt.scatter(x=to_plot[:, 0].cpu() * patch_size[0], y=to_plot[:, 1].cpu() * patch_size[1],
                                        color="red", s=2, alpha=0.6)
                            ax4 = plt.subplot(154)
                            gt = label_batch[0, :, :, 40].cpu().numpy()
                            plt.imshow(gt, cmap=plt.get_cmap('binary'))
                            plt.scatter(x=to_plot[:, 0].cpu() * patch_size[0], y=to_plot[:, 1].cpu() * patch_size[1],
                                        color="red", s=2, alpha=0.6)
                            # ax5 = plt.subplot(155)
                            # import pdb
                            # pdb.set_trace()
                            # highlighted_img = volume_batch[0, 0:1, :, :, 40].cpu().numpy().copy().repeat(3, axis = 0)
                            # highlighted_img[(gt[None,:] == 1).repeat(3, axis=0)][0] = 255
                            # plt.imshow(highlighted_img.transpose([1,2,0]))
                            # plt.scatter(x=to_plot[:, 0].cpu() * patch_size[0], y=to_plot[:, 1].cpu() * patch_size[1],
                            #             color="red", s=2, alpha=0.6)


                            writer.add_figure(
                                'point_coords',
                                fig,
                                global_step=iter_num,
                                close=False,
                                walltime=None)
                            # if iter_num >= 50:
                            #     import pdb
                            #     pdb.set_trace()





                image = uncertainty[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                # mask2 = (uncertainty > threshold).float()
                # image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # grid_image = make_grid(image, 5, normalize=True)
                # writer.add_image('train/mask', grid_image, iter_num)
                #####
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)


                with torch.no_grad():
                    unlabeled_output_prob = F.softmax(outputs[labeled_bs:], dim=1)

                # unlabeled_output_prob = output_prob
                image = torch.max(unlabeled_output_prob[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = torch.max(ema_output_prob[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Ema_predicted_label', grid_image, iter_num)

                with torch.no_grad():
                    for mc_sample_id in range(num_mc_samples):
                        unlabeled_ema_sample_prob = F.softmax(mc_samples_ema[mc_sample_id], dim=1)
                        image = torch.max(unlabeled_ema_sample_prob[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0,
                                                                                                      1).data.cpu().numpy()
                        image = utils.decode_seg_map_sequence(image)
                        grid_image = make_grid(image, 5, normalize=False)
                        writer.add_image('unlabel/unlabeled_ema_sample_{}_prob'.format(mc_sample_id), grid_image, iter_num)

                        unlabeled_sample_prob = F.softmax(mc_samples_unsupervised[mc_sample_id], dim=1)
                        image = torch.max(unlabeled_sample_prob[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0,
                                                                                                          1).data.cpu().numpy()
                        image = utils.decode_seg_map_sequence(image)
                        grid_image = make_grid(image, 5, normalize=False)
                        writer.add_image('unlabel/unlabeled_sample_{}_prob'.format(mc_sample_id), grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict(),
                            'ssn_rank': ssn_rank, 'max_iterations': max_iterations, 'bn_type': bn_type, 'baseline_noout': args.baseline_noout,
                            'head_normalization':args.head_normalization, 'head_layer_num': args.head_layer_num}, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict(),
                'ssn_rank': ssn_rank, 'max_iterations': max_iterations, 'bn_type': bn_type, 'baseline_noout':args.baseline_noout,
                'head_normalization':args.head_normalization, 'head_layer_num': args.head_layer_num}, save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
