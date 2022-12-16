import argparse

import logging
from collections import OrderedDict
import  time
import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
import torch.backends.cudnn as cudnn

import random
import numpy as np
import os
import shutil
import sys
from utils.semantic_dist_estimator import semantic_dist_estimator


import torchvision


from networks.stochastic_vnet import StochasticVNetSupCon
from dataloaders.pancreas import PancreasMasked as Pancreas
from dataloaders.la_heart import RandomCropMasked as RandomCrop, \
    CenterCropMasked as CenterCrop, RandomRotFlipMasked as RandomRotFlip, RandomFlipMasked as RandomFlip,\
    Random3DFlipMasked as Random3DFlip,\
    RandomAffineMasked as RandomAffine, GaussianBlurMasked as GaussianBlur,\
    NoAugMasked as NoAug,   \
    ToTensorMasked as ToTensor, LabeledBatchSampler, UnlabeledBatchSampler

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_3items/Pancreas-CT-training', help='Name of Experiment')
parser.add_argument('--image_list_path', type=str, default='pancreas_train.list', help='image_list_path')
parser.add_argument('--semantic_dist_dir', type=str, default='../data_3items/Pancreas-CT-semantic-dist', help='semantic_dist_dir')
parser.add_argument('--load_epoch_num', type=int, default=3000, help='load_epoch_num') # or 60 for upper bound
parser.add_argument('--load_model_name', type=str, default='exp_pancreas_254', help='load_model_name') # or 60 for upper bound


parser.add_argument('--exp', type=str,  default='pancreas_exp_semantic_dist_init', help='model_name')

parser.add_argument('--labeled_num', type=int,  default=12, help='labeled_num')
parser.add_argument('--total_num', type=int, default=12, help='total_num') # or 60 for upper bound
parser.add_argument('--max_iterations', type=int,  default=3000, help='maximum epoch number to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--extracted_feature_name', type=str, default='out_conv', help='extracted_feature_name')
parser.add_argument('--extracted_output_name', type=str, default='mean_l', help='extracted_output_name')
parser.add_argument('--feat_dist_save_name', type=str, default='feat_dist', help='feat_dist_save_name')
parser.add_argument('--out_dist_save_name', type=str, default='out_dist', help='out_dist_save_name')


args = parser.parse_args()
train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"
load_model_path = "../model/" + args.load_model_name + "/"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict




total_feat_out = {}
total_feat_in = {}

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
    modules = net.named_children()  #

    layers_name = list(net._modules.keys())
    layers = list(net._modules.values())

    def hook_fn_forward(module, input, output):
        layer = layers_name[np.argwhere([module == m for m in layers])[0, 0]]
        total_feat_out[layer] = output
        total_feat_in[layer] = input


    for name, module in modules:
        if  name == args.extracted_feature_name or name == args.extracted_output_name:
            module.register_forward_hook(hook_fn_forward)
        # module.register_backward_hook(hook_fn_backwar

    return net



if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("semantic_dist_init.trainer")


    feat_estimator = semantic_dist_estimator(feature_num=feature_num, num_classes=num_classes,
                                             feat_dist_save_name=args.feat_dist_save_name , out_dist_save_name=args.out_dist_save_name)
    out_estimator = semantic_dist_estimator(feature_num=num_classes, num_classes=num_classes,
                                            feat_dist_save_name=args.feat_dist_save_name, out_dist_save_name=args.out_dist_save_name)
    # create model
    model = load_1st_stage_model()
    torch.cuda.empty_cache()


    iteration = 0

    transforms_train = transforms.Compose([
        RandomCrop(patch_size),
        ToTensor(),
    ])
    db_train_labeled = Pancreas(base_dir=train_data_path,
                                split='train',
                                transform=transforms_train,
                                image_list_path=args.image_list_path)
    labeled_idxs = list(range(args.labeled_num))
    labeled_batch_sampler = LabeledBatchSampler(labeled_idxs, args.labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    labeledtrainloader = DataLoader(db_train_labeled, batch_sampler=labeled_batch_sampler, num_workers=4,
                                    pin_memory=True, worker_init_fn=worker_init_fn)
    max_epoch = args.max_iterations // len(labeledtrainloader) + 1
    print('max_epoch {}'.format(max_epoch))

    end = time.time()
    start_time = time.time()
    logging.info(">>>>>>>>>>>>>>>> Initialize semantic distributions >>>>>>>>>>>>>>>>")

    with torch.no_grad():

        for epoch_num in tqdm(range(max_epoch), ncols=70):
            time1 = time.time()
            for i_batch, labeled_sampled_batch in enumerate(labeledtrainloader):
                time2 = time.time()
                volume_batch, label_batch, sampling_mask_batch = labeled_sampled_batch['image'], labeled_sampled_batch['label'], labeled_sampled_batch['mask']
                volume_batch, label_batch, sampling_mask_batch = volume_batch.cuda(), label_batch.cuda(), sampling_mask_batch.cuda()

                res = model(volume_batch, sampling_mask_batch) # , mask=sampling_mask_batch
                features_labeled = total_feat_out[args.extracted_feature_name]
                outputs_labeled = total_feat_out[args.extracted_output_name]


                B, A, Hs, Ws, Ds = features_labeled.size()

                labeled_mask = label_batch.squeeze(0).long()
                labeled_mask = labeled_mask.contiguous().view(B * Hs * Ws * Ds, )

                features_labeled = features_labeled.permute(0, 2, 3, 4, 1).contiguous().view(B * Hs * Ws * Ds, A)
                # update feature-level statistics
                feat_estimator.update(features=features_labeled.detach(), labels=labeled_mask)
                outputs_labeled = outputs_labeled.permute(0, 2, 3, 4, 1).contiguous().view(B * Hs * Ws * Ds, num_classes)
                # update output-level statistics
                out_estimator.update(features=outputs_labeled.detach(), labels=labeled_mask)
        feat_estimator.save(name=args.feat_dist_save_name + '.pth', save_dir=args.semantic_dist_dir)
        out_estimator.save(name=args.out_dist_save_name + '.pth', save_dir=args.semantic_dist_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logging.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_time / args.max_iterations
        )
    )
