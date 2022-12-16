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

from networks.stochastic_vnet import StochasticVNetSupCon
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
parser.add_argument('--cross_image_sampling', type=str2bool,  default=False, help='cross_image_sampling')
parser.add_argument('--head_normalization', type=str,  default='none', help='head_normalization')
parser.add_argument('--head_layer_num', type=int,  default=3, help='head_layer_num')
## parameters for pseudo contrastive loss
parser.add_argument('--cross_unsup_image_contrast', type=str2bool,  default=True, help='cross_unsup_image_contrast')
parser.add_argument('--cross_unsup_image_sampling', type=str2bool,  default=False, help='cross_unsup_image_sampling')
parser.add_argument('--unsup_cont_weight', type=float, default=1, help='unsup_cont_weight')
parser.add_argument('--unsup_cont_rampup_scheme', type=str,  default='sigmoid_rampup', help='unsup_cont_rampup_scheme')
parser.add_argument('--unsup_cont_rampup', type=float,  default=40, help='sup_cont_rampdown')
parser.add_argument('--random_unsup_sampled_num', type=int,  default=1000, help='random_unsup_sampled_num')

### costs
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
parser.add_argument('--ssn_rank', type=int,  default=10, help='ssn rank')
parser.add_argument('--pairwise_dist_metric', type=str,  default='dice', help='ssn rank')
parser.add_argument('--oracle_checking', type=str2bool,  default=False, help='oracle_checking')
parser.add_argument('--with_uncertainty_mask', type=str2bool,  default=False, help='with_uncertainty_mask')

# ablation study parameters for generalized energy distance
parser.add_argument('--weight_div1', type=float,  default=1, help='weight for diversity1 in loss function')
parser.add_argument('--weight_div2', type=float,  default=1, help='weight for diversity2 in loss function')

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


args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

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

# ablation study parameters for generalized energy distance
weight_div1 = args.weight_div1
weight_div2 = args.weight_div2
# region level generalized energy distance

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

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # print(ema_model.state_dict()['block_one.conv.1.running_var'])

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

def oracle_check(output_logits, label, mc_samples ):
    b,c,h,w,d = output_logits.size()
    loss_seg_mc_intg = losses.loss_mc_integral_given_samples(mc_samples, label, num_mc_samples) / (h * w * d)
    output_prob = F.softmax(output_logits, dim=1)
    loss_seg_dice = losses.dice_loss(output_prob[:, 1, :, :, :], label == 1)
    supervised_loss = 0.5 * (loss_seg_mc_intg + loss_seg_dice)

    _, output = torch.max(output_logits, dim=1)
    dice_coefficient = losses.dice_loss(output, label)
    return supervised_loss, loss_seg_mc_intg, loss_seg_dice, dice_coefficient


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
            consistency_dist_each_batch_sample.append(
                    losses.generalised_energy_distance(stochastic_samples_logits[:, batch_sample_id],
                                                       ema_stochastic_samples_logits[:, batch_sample_id],
                                                       pairwise_dist_metric, weight_div1, weight_div2,)[0])

        else:
            # consistency_dist_each_batch_sample.append(
            #     losses.generalised_energy_distance(stochastic_samples_pred, ema_stochastic_samples_pred,pairwise_dist_metric)[0])
            consistency_dist_each_batch_sample.append(
                    losses.generalised_energy_distance(stochastic_samples_pred, ema_stochastic_samples_pred,
                                                       pairwise_dist_metric, weight_div1, weight_div2)[0])
    consistency_dist = torch.stack(consistency_dist_each_batch_sample).mean()
    consistency_loss = consistency_weight * consistency_dist
    return consistency_loss, consistency_dist


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


    def create_model(ema=False):
        # Network definition
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
    labeled_batch_sampler = LabeledBatchSampler(labeled_idxs, labeled_bs,args.sampler_fixed)
    unlabeled_batch_sampler = UnlabeledBatchSampler(unlabeled_idxs, batch_size - labeled_bs,args.sampler_fixed)


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


            # 1: accepts returned features
            outputs, features, states = model(volume_batch, sampling_mask_batch)

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
            supervised_loss, loss_seg_mc_intg, loss_seg_dice = get_superivsed_loss(outputs[:labeled_bs],
                                                                                       label_batch[:labeled_bs],
                                                                                       mc_samples_supervised, with_dice)

            #  supervised contrastive learning
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
                            # how to take pixels
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

                            # how to take pixels
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


            else:
                loss_supervised_contrastive = torch.zeros([1])
                running_sup_cont_weight = 0

            if oracle_checking:
                with torch.no_grad(): # oracle testing
                    supervised_loss_student_batch, loss_student_mc_intg, loss_student_dice, dice_coeff_student = oracle_check(outputs,label_batch,mc_samples)
                    mc_samples_teacher_batch = losses.fixed_re_parametrization_trick(teacher_states_batch['distribution'], num_mc_samples)
                    supervised_loss_teacher_batch, loss_teacher_mc_intg, loss_teacher_dice, dice_coeff_teacher = oracle_check(teacher_output_batch, label_batch, mc_samples_teacher_batch)


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
                consistency_loss = consistency_loss
            elif args.consistency_type == 'None':
                consistency_loss, consistency_dist = torch.tensor(0), torch.tensor(0)


            loss = supervised_loss + consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

            writer.add_scalar('loss/loss_supervised_contrastive', loss_supervised_contrastive, iter_num)
            writer.add_scalar('loss/running_sup_cont_weight', running_sup_cont_weight, iter_num)


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


            writer.add_scalar('train/consistency_weight_uncertainty', consistency_weight_uncertainty, iter_num)
            writer.add_scalar('train/cov_diag_consistency_dist', cov_diag_consistency_dist, iter_num)
            writer.add_scalar('train/cov_factor_consistency_dist', cov_factor_consistency_dist, iter_num)
            writer.add_scalar('train/cov_diag_consistency_loss', cov_factor_consistency_loss, iter_num)
            writer.add_scalar('train/cov_factor_consistency_loss', cov_factor_consistency_loss, iter_num)

            basic_info = 'iteration %d : loss : %f supervised_loss: %f, loss_seg_mc_intg: %f, loss_seg_dice: %f,  loss_supervised_contrastive: %f, running_sup_cont_weight: %f, cons_dist: %f, loss_weight: %f ' % \
                             (iter_num, loss.item(), supervised_loss.item(), loss_seg_mc_intg.item(),
                              loss_seg_dice.item(), loss_supervised_contrastive.item(), running_sup_cont_weight,
                              consistency_dist.item(),
                              consistency_weight)
            logging.info(basic_info )

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

                image = uncertainty[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

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
                            'ssn_rank': ssn_rank, 'max_iterations': max_iterations, 'bn_type': bn_type,
                            'head_normalization':args.head_normalization, 'head_layer_num': args.head_layer_num}, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict(),
                'ssn_rank': ssn_rank, 'max_iterations': max_iterations, 'bn_type': bn_type,
                'head_normalization':args.head_normalization, 'head_layer_num': args.head_layer_num}, save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
