import os
import argparse
import torch
from networks.stochastic_vnet import StochasticVNetSupCon
from test_util import test_all_case_masked as test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_3items/Pancreas-CT-test/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--iter', type=int,  default=5000, help='model iteration')
parser.add_argument('--ssn_rank', type=int,  default=10, help='ssn rank')
parser.add_argument('--maxiter', type=int,  default=6000, help='model iteration')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2
with open(FLAGS.root_path + '/../pancreas_test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    checkpoint = torch.load(save_mode_path)
    ssn_rank = checkpoint['ssn_rank']
    bn_type = checkpoint['bn_type']
    head_normalization = checkpoint['head_normalization']
    head_layer_num = checkpoint['head_layer_num']
    net = StochasticVNetSupCon(input_channels=1, num_classes=num_classes, normalization=bn_type, has_dropout=False, rank=ssn_rank,
                                   head_normalization=head_normalization, head_layer_num=head_layer_num).cuda()

    net.load_state_dict(checkpoint['model'])

    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    for iter_id in range(FLAGS.maxiter // 1000):
        iteration = (iter_id + 1) * 1000
        print('performance of iteration {}'.format(iteration))
        metric = test_calculate_metric(iteration)
        print(metric)

