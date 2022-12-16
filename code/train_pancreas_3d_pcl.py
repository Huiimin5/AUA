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
from utils.semantic_dist_estimator import semantic_dist_estimator
from dataloaders.la_heart import LAHeartMasked as LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import torchio
from dataloaders.pancreas import PancreasMasked as Pancreas
from dataloaders.la_heart import RandomCropMasked as RandomCrop, \
    CenterCropMasked as CenterCrop, RandomRotFlipMasked as RandomRotFlip, RandomFlipMasked as RandomFlip,\
    Random3DFlipMasked as Random3DFlip,\
    RandomAffineMasked as RandomAffine, GaussianBlurMasked as GaussianBlur,\
    NoAugMasked as NoAug,   \
    ToTensorMasked as ToTensor, LabeledBatchSampler, UnlabeledBatchSampler
from val_3D import test_batch
def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
# todo: adapt from .py
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_3items/Pancreas-CT-training', help='Name of Experiment')
parser.add_argument('--image_list_path', type=str, default='pancreas_train.list', help='image_list_path')
parser.add_argument('--semantic_dist_dir', type=str, default='../data_3items/Pancreas-CT-semantic-dist', help='semantic_dist_dir')


parser.add_argument('--exp', type=str,  default='pancreas_exp_000', help='model_name')
parser.add_argument('--labeled_num', type=int,  default=12, help='labeled_num')
parser.add_argument('--total_num', type=int, default=60, help='total_num') # or 60 for upper bound
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')


# 1st stage model
parser.add_argument('--load_model_name', type=str, default='exp_pancreas_254', help='load_model_name') # or 60 for upper bound
parser.add_argument('--load_epoch_num', type=int, default=3000, help='load_epoch_num') # or 60 for upper bound

### 2nd stage model creation
parser.add_argument('--bn_type', type=str,  default='batchnorm', help='bn_type')
parser.add_argument('--head_normalization', type=str,  default='none', help='head_normalization')
parser.add_argument('--head_layer_num', type=int,  default=3, help='head_layer_num')
parser.add_argument('--head_pos_id', type=int,  default=0, help='head_pos_id')
parser.add_argument('--ssn_rank', type=int,  default=10, help='ssn rank')
parser.add_argument('--ORACLE', type=str2bool,  default=False, help='oracle')

### costs
parser.add_argument('--with_dice', type=str2bool,  default=True, help='with_dice loss')
parser.add_argument('--LAMBDA_FEAT', type=float,  default=0, help='LAMBDA_FEAT')
parser.add_argument('--LAMBDA_FEAT_labeled', type=float,  default=1, help='LAMBDA_FEAT_labeled')
parser.add_argument('--LAMBDA_FEAT_unlabeled', type=float,  default=1, help='LAMBDA_FEAT_unlabeled')

parser.add_argument('--LAMBDA_OUT', type=float,  default=0, help='LAMBDA_OUT')
parser.add_argument('--LAMBDA_OUT_labeled', type=float,  default=1, help='LAMBDA_OUT_labeled')
parser.add_argument('--LAMBDA_OUT_unlabeled', type=float,  default=1, help='LAMBDA_OUT_unlabeled')

parser.add_argument('--LAMBDA_PSEUDO', type=float,  default=1, help='LAMBDA_PSEUDO')
parser.add_argument('--temp', type=float, default=100, help='temperature for loss function')
parser.add_argument('--estimator_starts_iter', type=int,  default=3000, help='estimator_starts_iter')

parser.add_argument('--extracted_feature_name', type=str, default='out_conv', help='extracted_feature_name')
parser.add_argument('--extracted_output_name', type=str, default='mean_l', help='extracted_output_name')
parser.add_argument('--IGNORE_LABEL', type=int,  default=255, help='IGNORE_LABEL')
parser.add_argument('--feat_dist_save_name', type=str, default='feat_dist', help='feat_dist_save_name')
parser.add_argument('--out_dist_save_name', type=str, default='out_dist', help='out_dist_save_name')

parser.add_argument('--rampup_total_epoch', type=float,  default=40.0, help='rampup_total_epoch')
parser.add_argument('--LAMBDA_FEAT_ramp_up_scheduler', type=str, default='', help='LAMBDA_FEAT_ramp_up_scheduler')
parser.add_argument('--LAMBDA_OUT_ramp_up_scheduler', type=str, default='', help='LAMBDA_OUT_ramp_up_scheduler')



# data loader augmentation for labeled
parser.add_argument('--data_loader_labeled_aug', type=str, default='RandomRotFlip', help='data_loader_labeled_aug')
# data loader augmentation for unlabeled
parser.add_argument('--unlabeled_aug_with_resize', type=str2bool, default=False, help='unlabeled_aug_with_resize')
parser.add_argument('--unlabeled_aug_with_gaussian_blur', type=str2bool, default=False, help='unlabeled_aug_with_gaussian_blur')
parser.add_argument('--unlabeled_aug_with_rotationflip', type=str2bool, default=False, help='unlabeled_aug_with_rotationflip')
parser.add_argument('--unlabeled_aug_with_flip', type=str2bool, default=False, help='unlabeled_aug_with_flip')
parser.add_argument('--unlabeled_augT_with_gaussian_blur', type=str2bool, default=False, help='unlabeled_augT_with_gaussian_blur')
# for debug
parser.add_argument('--transform_fixed', type=str2bool, default=False, help='')
parser.add_argument('--sampler_fixed', type=str2bool, default=False, help='')


args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"
load_model_path = "../model/" + args.load_model_name + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
with_dice = args.with_dice
bn_type = args.bn_type
# ablation study of data_loader_labeled_aug
data_loader_labeled_aug = args.data_loader_labeled_aug

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)
feature_num = 16  # todo: automatically

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

def get_ramping_up_weight(epoch, ramp_up_scheduler):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if ramp_up_scheduler == 'sigmoid_rampup':
        return ramps.sigmoid_rampup(epoch, args.rampup_total_epoch)
    elif ramp_up_scheduler == 'linear_rampup':
        return ramps.linear_rampup(epoch, args.rampup_total_epoch)
    elif ramp_up_scheduler == 'log_rampup':
        return ramps.log_rampup(epoch, args.rampup_total_epoch)
    elif ramp_up_scheduler == 'exp_rampup':
        return ramps.exp_rampup(epoch, args.rampup_total_epoch)
    else:
        return 1

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
                                   rank=args.ssn_rank, head_normalization=args.head_normalization,
                                   head_layer_num=args.head_layer_num)

        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    def load_1st_stage_model():
        # Network definition
        save_mode_path = os.path.join(load_model_path, 'iter_' + str(args.load_epoch_num) + '.pth')
        checkpoint = torch.load(save_mode_path, map_location=torch.device('cpu'))
        ssn_rank = checkpoint['ssn_rank']
        bn_type = checkpoint['bn_type']
        baseline_noout = checkpoint['baseline_noout']
        head_normalization = checkpoint['head_normalization']
        head_layer_num = checkpoint['head_layer_num']
        net = StochasticVNetSupCon(input_channels=1, num_classes=num_classes, normalization=bn_type, has_dropout=False,
                                   rank=ssn_rank,
                                   head_normalization=head_normalization, head_layer_num=head_layer_num).cuda()

        net.load_state_dict(checkpoint['model'])

        print("init weight from {}".format(save_mode_path))
        net.eval()

        return net


    model = create_model()
    # register hook for feature extraction
    modules = model.named_children()  #
    layers_name = list(model._modules.keys())
    layers = list(model._modules.values())
    total_feat_out = {}
    total_feat_in = {}
    def hook_fn_forward(module, input, output):
        layer = layers_name[np.argwhere([module == m for m in layers])[0, 0]]
        total_feat_out[layer] = output
        total_feat_in[layer] = input

    for name, module in modules:
        if name == args.extracted_feature_name or name == args.extracted_output_name:
            module.register_forward_hook(hook_fn_forward)

    model_1st_stage = load_1st_stage_model() # todo: model creation and other assertions to make sure pseudo labels generation as expected
    model_1st_stage.eval()
    # ema_model = create_model(ema=True)

    """data loader and sampler for labeled training"""
    transforms_train = transforms.Compose([
        RandomCrop(patch_size, args.transform_fixed),
        ToTensor(),
    ])
    if data_loader_labeled_aug ==  'RandomRotFlip':
        transforms_train.transforms.insert(0, RandomRotFlip(args.transform_fixed))
    elif data_loader_labeled_aug == 'RandomFlip':
        transforms_train.transforms.insert(0, RandomFlip(args.transform_fixed))
    elif data_loader_labeled_aug == 'Random3DFlip':
        transforms_train.transforms.insert(0, Random3DFlip(args.transform_fixed))
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
    if args.unlabeled_aug_with_gaussian_blur:
        transforms_train_unlabeled.transforms.insert(0, GaussianBlur())
    if args.unlabeled_aug_with_resize:
        transforms_train_unlabeled.transforms.insert(0, RandomAffine())
    if args.unlabeled_aug_with_rotationflip:
        assert not args.unlabeled_aug_with_flip
        transforms_train_unlabeled.transforms.insert(0, RandomRotFlip(args.transform_fixed))
    if args.unlabeled_aug_with_flip:
        transforms_train_unlabeled.transforms.insert(0, RandomFlip(args.transform_fixed))

    db_train_unlabeled = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform=transforms_train_unlabeled,
                                  image_list_path=args.image_list_path)


    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_num))
    labeled_batch_sampler = LabeledBatchSampler(labeled_idxs, labeled_bs, args.sampler_fixed)
    unlabeled_batch_sampler = UnlabeledBatchSampler(unlabeled_idxs, batch_size - labeled_bs, args.sampler_fixed)

    def worker_init_fn(worker_id):
        if args.sampler_fixed:
            random.seed(args.seed)
        else:
            random.seed(args.seed+worker_id)
    labeledtrainloader = DataLoader(db_train_labeled, batch_sampler=labeled_batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    unlabeledtrainloader = DataLoader(db_train_unlabeled, batch_sampler=unlabeled_batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    # init loss
    pcl_criterion = losses.PixelContrastiveLoss(args.temp, args.IGNORE_LABEL)

    # load semantic distributions
    logging.info(">>>>>>>>>>>>>>>> Load semantic distributions >>>>>>>>>>>>>>>>")
    feat_estimator = semantic_dist_estimator(feature_num=feature_num, num_classes=num_classes,semantic_dist_dir=args.semantic_dist_dir,
                                             feat_dist_save_name=args.feat_dist_save_name,out_dist_save_name=args.out_dist_save_name)
    if args.LAMBDA_OUT != 0:
        out_estimator = semantic_dist_estimator(feature_num=num_classes, num_classes=num_classes,semantic_dist_dir=args.semantic_dist_dir,
                                                feat_dist_save_name=args.feat_dist_save_name,out_dist_save_name=args.out_dist_save_name)



    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(labeledtrainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(labeledtrainloader)+1
    lr_ = base_lr

    # unlabeled_iter = iter(unlabeledtrainloader)

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, (labeled_sampled_batch, unlabeled_sampled_batch) in enumerate(zip(labeledtrainloader, unlabeledtrainloader)):
            time2 = time.time()
            # unlabeled_sampled_batch = next(unlabeled_iter)
            unlabeled_volume_batch, unlabel_label_batch, unlabeled_mask_batch = unlabeled_sampled_batch['image'], \
                                                                                unlabeled_sampled_batch['label'], \
                                                                                unlabeled_sampled_batch['mask']

            volume_batch, label_batch, sampling_mask_batch = torch.cat([labeled_sampled_batch['image'], unlabeled_volume_batch]), \
                                                             torch.cat([labeled_sampled_batch['label'], unlabel_label_batch]), \
                                                             torch.cat([labeled_sampled_batch['mask'], unlabeled_mask_batch])
            # push to gpu
            unlabeled_volume_batch, unlabel_label_batch, unlabeled_mask_batch = unlabeled_volume_batch.cuda(), unlabel_label_batch.cuda(), unlabeled_mask_batch.cuda()
            volume_batch, label_batch, sampling_mask_batch = volume_batch.cuda(), label_batch.cuda(), sampling_mask_batch.cuda()

            outputs_, _, _ = model(volume_batch, sampling_mask_batch)
            outputs, features = total_feat_out[args.extracted_output_name], total_feat_out[args.extracted_feature_name]
            outputs_labeled, outputs_unlabeled = outputs[:labeled_bs], outputs[labeled_bs:]
            features_labeled, features_unlabeled = features[:labeled_bs], features[labeled_bs:]
            outputs_unlabeled_pseudo, _, _ = model_1st_stage(unlabeled_volume_batch, unlabeled_mask_batch)

            output_prob = F.softmax(outputs, dim=1)
            # monitor the performance of each branch
            output_pred = torch.argmax(output_prob, dim=1)
            training_metric = test_batch(output_pred.cpu().data.numpy(),label_batch.cpu().data.numpy(), num_classes=num_classes)

            # supervision loss
            supervised_loss, loss_seg_mc_intg, loss_seg_dice = get_supervised_loss_no_uncertainty(outputs_labeled, label_batch[:labeled_bs])


            # source mask: downsample the ground-truth label
            B, A, Hs, Ws, Ds = features_labeled.size()
            # if args.head_pos_id != 0:
            #     labeled_mask = F.interpolate(label_batch[:labeled_bs].unsqueeze(0).float(), size=(Hs, Ws, Ds), mode='nearest').squeeze(0).long()
            labeled_mask = label_batch[:labeled_bs].contiguous().view(B * Hs * Ws * Ds, )
            assert not labeled_mask.requires_grad
            # target mask: constant threshold -- cfg.SOLVER.THRESHOLD
            # _, _, Ht, Wt = tgt_feat.size()
            unlabeled_out_maxvalue, unlabeled_mask = torch.max(outputs_unlabeled_pseudo, dim=1)
            if args.ORACLE:
                unlabeled_mask = label_batch[labeled_bs:]

            if args.LAMBDA_PSEUDO > 0:
                loss_pseudo, loss_seg_mc_intg_ul, loss_seg_dice_ul = get_supervised_loss_no_uncertainty(outputs_unlabeled, unlabeled_mask)
            else:
                loss_pseudo, loss_seg_mc_intg_ul, loss_seg_dice_ul  = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            # for i in range(cfg.MODEL.NUM_CLASSES):
            #     tgt_mask[(tgt_out_maxvalue < cfg.SOLVER.DELTA) * (tgt_mask == i)] = 255
            unlabeled_mask = unlabeled_mask.contiguous().view(B * Hs * Ws * Ds, )
            assert not unlabeled_mask.requires_grad

            features_labeled = features_labeled.permute(0, 2, 3, 4, 1).contiguous().view(B * Hs * Ws * Ds, A)
            features_unlabeled = features_unlabeled.permute(0, 2, 3, 4, 1).contiguous().view(B * Hs * Ws * Ds, A)

            # update feature-level statistics
            if iter_num >= args.estimator_starts_iter:
                feat_estimator.update(features=features_labeled.detach(), labels=labeled_mask)
            # contrastive loss on both domains
            if args.LAMBDA_FEAT != 0:
                if args.LAMBDA_FEAT_labeled == 1 and args.LAMBDA_FEAT_unlabeled == 1:
                    loss_feat = pcl_criterion(Mean=feat_estimator.Mean.detach(),
                                              CoVariance=feat_estimator.CoVariance.detach(),
                                              feat=features_labeled,
                                              labels=labeled_mask) \
                                + pcl_criterion(Mean=feat_estimator.Mean.detach(),
                                                CoVariance=feat_estimator.CoVariance.detach(),
                                                feat=features_unlabeled,
                                                labels=unlabeled_mask)
                elif args.LAMBDA_FEAT_labeled == 1 and args.LAMBDA_FEAT_unlabeled == 0:
                    loss_feat = pcl_criterion(Mean=feat_estimator.Mean.detach(),
                                              CoVariance=feat_estimator.CoVariance.detach(),
                                              feat=features_labeled,
                                              labels=labeled_mask)
                elif args.LAMBDA_FEAT_labeled == 0 and args.LAMBDA_FEAT_unlabeled == 1:
                    loss_feat = pcl_criterion(Mean=feat_estimator.Mean.detach(),
                                                CoVariance=feat_estimator.CoVariance.detach(),
                                                feat=features_unlabeled,
                                                labels=unlabeled_mask)

            else:
                loss_feat = torch.tensor(0)

            if args.LAMBDA_OUT != 0:
                outputs_labeled = outputs_labeled.permute(0, 2, 3, 4, 1).contiguous().view(B * Hs * Ws * Ds, num_classes)
                outputs_unlabeled = outputs_unlabeled.permute(0, 2, 3, 4, 1).contiguous().view(B * Hs * Ws * Ds, num_classes)

                # update output-level statistics
                if iter_num >= args.estimator_starts_iter:
                    out_estimator.update(features=outputs_labeled.detach(), labels=labeled_mask)

                # the proposed contrastive loss on prediction map
                if args.LAMBDA_OUT_labeled == 1 and args.LAMBDA_OUT_unlabeled == 1:
                    loss_out = pcl_criterion(Mean=out_estimator.Mean.detach(),
                                             CoVariance=out_estimator.CoVariance.detach(),
                                             feat=outputs_labeled,
                                             labels=labeled_mask) \
                               + pcl_criterion(Mean=out_estimator.Mean.detach(),
                                               CoVariance=out_estimator.CoVariance.detach(),
                                               feat=outputs_unlabeled,
                                               labels=unlabeled_mask)
                elif args.LAMBDA_OUT_labeled == 1 and args.LAMBDA_OUT_unlabeled == 0:
                    loss_out = pcl_criterion(Mean=out_estimator.Mean.detach(),
                                             CoVariance=out_estimator.CoVariance.detach(),
                                             feat=outputs_labeled,
                                             labels=labeled_mask)
                elif args.LAMBDA_OUT_labeled == 0 and args.LAMBDA_OUT_unlabeled == 1:
                    loss_out = pcl_criterion(Mean=out_estimator.Mean.detach(),
                                               CoVariance=out_estimator.CoVariance.detach(),
                                               feat=outputs_unlabeled,
                                               labels=unlabeled_mask)
            else:
                loss_out = torch.tensor(0)

            running_LAMBDA_FEAT = args.LAMBDA_FEAT * get_ramping_up_weight(iter_num // 150, args.LAMBDA_FEAT_ramp_up_scheduler)
            running_LAMBDA_OUT = args.LAMBDA_OUT * get_ramping_up_weight(iter_num // 150, args.LAMBDA_OUT_ramp_up_scheduler)

            loss = supervised_loss + running_LAMBDA_FEAT * loss_feat + args.LAMBDA_OUT * loss_out + args.LAMBDA_PSEUDO * loss_pseudo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            # writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            # writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg_mc_intg', loss_seg_mc_intg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            basic_info = 'iteration %d : loss : %f supervised_loss: %f, loss_seg_mc_intg: %f, loss_seg_dice: %f, '  % \
                         (iter_num, loss.item(), supervised_loss.item(), loss_seg_mc_intg.item(), loss_seg_dice.item())
            if args.LAMBDA_FEAT:
                basic_info += 'loss_feat: %f, running_LAMBDA_FEAT: %f, ' % (loss_feat.item(), running_LAMBDA_FEAT)
            if args.LAMBDA_OUT != 0:
                basic_info += 'loss_out: %f, running_LAMBDA_OUT: %f, ' % (loss_out.item(), running_LAMBDA_OUT)
            if args.LAMBDA_PSEUDO != 0:
                basic_info += 'loss_pseudo: %f, loss_seg_mc_intg_ul: %f, loss_seg_dice_ul: %f, ' % (loss_pseudo.item(), loss_seg_mc_intg_ul.item(), loss_seg_dice_ul.item())
            logging.info(basic_info)

            for i in range(num_classes-1):
                writer.add_scalar('train_evaluator/dice_class{}'.format(i+1), training_metric[i][0], iter_num)
                writer.add_scalar('train_evaluator/hd95_class{}'.format(i+1), training_metric[i][1], iter_num)

            if iter_num % 50 == 0:
                labeled_output_prob = output_prob[:labeled_bs]
                unlabeled_output_prob = F.softmax(outputs[labeled_bs:], dim=1)

                labeled_output_pred = torch.argmax(labeled_output_prob, dim=1)
                unlabeled_output_pred = torch.argmax(unlabeled_output_prob, dim=1)

                metric_labeled = test_batch(labeled_output_pred.cpu().data.numpy(),
                                            label_batch[:labeled_bs].cpu().data.numpy(), num_classes=num_classes)
                metric_unlabeled = test_batch(unlabeled_output_pred.cpu().data.numpy(),
                                              label_batch[labeled_bs:batch_size].cpu().data.numpy(),
                                              num_classes=num_classes)
                for i in range(num_classes - 1):
                    writer.add_scalars('train_evaluator/dice_class{}'.format(i+1),
                                       {'labeled': metric_labeled[i][0], 'unlabeled': metric_unlabeled[i][0]}, iter_num)
                    writer.add_scalars('train_evaluator/hd95_class{}'.format(i+1),
                                       {'labeled': metric_labeled[i][1], 'unlabeled': metric_unlabeled[i][1]}, iter_num)

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

                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)

                with torch.no_grad():
                    unlabeled_output_prob = F.softmax(outputs[labeled_bs:], dim=1)

                # unlabeled_output_prob = output_prob
                image = torch.max(unlabeled_output_prob[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0,
                                                                                              1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

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
                torch.save({'model': model.state_dict(),
                            'ssn_rank': args.ssn_rank, 'max_iterations': max_iterations, 'bn_type': bn_type,
                            'head_normalization': args.head_normalization, 'head_layer_num': args.head_layer_num},
                           save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save({'model': model.state_dict(),
                'ssn_rank': args.ssn_rank, 'max_iterations': max_iterations, 'bn_type': bn_type,
                'head_normalization': args.head_normalization, 'head_layer_num': args.head_layer_num},
               save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
