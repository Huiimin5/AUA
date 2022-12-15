import torch
from torch import nn
import torch.nn.functional as F
from dropblock import DropBlock2D, DropBlock3D

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
class HeadConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(HeadConvBlock, self).__init__()

        ops = []

        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 1))
            if i != n_stages - 1:
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
                elif normalization == 'groupnorm':
                    ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                elif normalization == 'instancenorm':
                    ops.append(nn.InstanceNorm3d(n_filters_out))
                elif normalization != 'none':
                    assert False
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
import numpy as np

class HeadUnsamplingConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(HeadUnsamplingConvBlock, self).__init__()

        ops = []
        n_upsampling_stage = int(np.log2(stride))
        if n_upsampling_stage > n_stages:
            n_stages = n_upsampling_stage

        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            if i < n_upsampling_stage:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
                ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=3, padding=1))
            else:
                ops.append(nn.Conv3d(input_channel, n_filters_out, 1))

            # todo1: upsampling
            # todo2 ops.append(nn.Conv3d(input_channel, n_filters_out, 1))
            if i != n_stages - 1:
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
                elif normalization == 'groupnorm':
                    ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                elif normalization == 'instancenorm':
                    ops.append(nn.InstanceNorm3d(n_filters_out))
                elif normalization != 'none':
                    assert False
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
class BeforeHeadUnsamplingConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(BeforeHeadUnsamplingConvBlock, self).__init__()

        ops = []
        n_upsampling_stage = int(np.log2(stride))
        if n_upsampling_stage > n_stages:
            n_stages = n_upsampling_stage

        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
                output_channel = input_channel // 2
            elif i < n_stages - 1:
                input_channel = output_channel
                output_channel = input_channel // 2
            else:
                input_channel = output_channel
                output_channel = n_filters_out

            if i < n_upsampling_stage:
                ops.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False))
                ops.append(nn.Conv3d(input_channel, output_channel, kernel_size=3, padding=1))
            else:
                ops.append(nn.Conv3d(input_channel, output_channel, 1))

            # todo1: upsampling
            # todo2 ops.append(nn.Conv3d(input_channel, n_filters_out, 1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

from torch.distributions.uniform import Uniform

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out
    def get_output_size(self,patch_size):
        return patch_size



    def forward(self, input, turnoff_drop=False, noise_weight=None,uniform_range=None, num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(features[layer_id].device).unsqueeze(0) # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out

class VNet_noout(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_noout, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        # self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        # out = self.out_conv(x9)
        out = x9
        return out
    def get_output_size(self,patch_size):
        return patch_size



    def forward(self, input, turnoff_drop=False, noise_weight=None,uniform_range=None, num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(features[layer_id].device).unsqueeze(0) # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out
# net before stochastic part
class VNetSupCon(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,head_layer_num=3, head_normalization='none'):
        super(VNetSupCon, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.head = HeadConvBlock(head_layer_num, n_classes, n_classes, normalization=head_normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        proj_features_unnorm = self.head(out)
        proj_features = F.normalize(proj_features_unnorm, dim=1)
        return out, proj_features

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, proj_features = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features

class VNetSupConMultiScale(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 head_multi_scale_num=1, head_layer_num=3, head_normalization='none'):
        super(VNetSupConMultiScale, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        # self.head = HeadConvBlock(head_layer_num, n_classes, n_classes, normalization=head_normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.head_multi_scale_num = head_multi_scale_num
        all_head_multi_scale = []
        for head_multi_scale_id in range(self.head_multi_scale_num):
            if head_multi_scale_id == 0:
                all_head_multi_scale.append(HeadConvBlock(head_layer_num, n_classes, n_classes, normalization=head_normalization))
            else:
                head_filter_num = n_filters * (2 ** head_multi_scale_id)
                current_stride = 2 ** head_multi_scale_id
                all_head_multi_scale.append(HeadUnsamplingConvBlock(head_layer_num, head_filter_num, head_filter_num,
                                                                    stride = current_stride, normalization=head_normalization))
        self.head_multi_scale = nn.ModuleList(all_head_multi_scale)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)

        dec_features_to_proj = [out, x8, x7, x6, x5]
        proj_features_multi_scale = []
        for head_multi_scale_id in range(self.head_multi_scale_num):
            current_head = self.head_multi_scale[head_multi_scale_id]
            proj_features_unnorm = current_head(dec_features_to_proj[head_multi_scale_id])
            proj_features = F.normalize(proj_features_unnorm, dim=1)
            proj_features_multi_scale.append(proj_features)

        return out, proj_features_multi_scale

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, proj_features_multi_scale = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features_multi_scale

class VNetSupConMultiScaleFusion(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 head_multi_scale_num=1, head_layer_num=3, head_normalization='none'):
        super(VNetSupConMultiScaleFusion, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        #
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.head_multi_scale_num = head_multi_scale_num
        all_upsampling_multi_scale = []
        for upsampling_multi_scale_id in range(1, self.head_multi_scale_num):
            head_filter_num_in = n_filters * (2 ** upsampling_multi_scale_id)
            head_filter_num_out = n_filters
            current_stride = 2 ** upsampling_multi_scale_id
            all_upsampling_multi_scale.append(BeforeHeadUnsamplingConvBlock(0, head_filter_num_in, head_filter_num_out,stride = current_stride, normalization=head_normalization))
        self.upsampling_multi_scale = nn.ModuleList(all_upsampling_multi_scale)
        self.head_fusion = HeadConvBlock(head_layer_num, n_classes + head_filter_num_out * (self.head_multi_scale_num-1), n_classes, normalization=head_normalization)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)

        dec_features_to_proj = [out, x8, x7, x6, x5]
        upsampled_features_multi_scale = [nn.ReLU(inplace=True)(out)]
        for head_multi_scale_id in range(1, self.head_multi_scale_num):
            upsampled_features = self.upsampling_multi_scale[head_multi_scale_id-1](dec_features_to_proj[head_multi_scale_id])
            upsampled_features_multi_scale.append(upsampled_features)
        upsampled_features_multi_scale_ = torch.cat(upsampled_features_multi_scale, dim=1)
        proj_features_unnorm = self.head_fusion(upsampled_features_multi_scale_)
        proj_features = F.normalize(proj_features_unnorm, dim=1)

        return out, proj_features

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, proj_features_multi_scale_fusion = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features_multi_scale_fusion


class VNetSupCon_noout(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 head_layer_num=3, head_normalization='none'):
        super(VNetSupCon_noout, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        # self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.head = HeadConvBlock(head_layer_num, n_filters, n_filters, normalization=head_normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        # out = self.out_conv(x9)
        proj_features_unnorm = self.head(x9)
        proj_features = F.normalize(proj_features_unnorm, dim=1)
        return x9, proj_features

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, proj_features = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
class VNetSupCon_head_before_out(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 head_layer_num=3, head_normalization='none'):
        super(VNetSupCon_head_before_out, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.head = HeadConvBlock(head_layer_num, n_filters, n_filters, normalization=head_normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        #
        proj_features_unnorm = self.head(x9)
        proj_features = F.normalize(proj_features_unnorm, dim=1)
        out = self.out_conv(x9)
        return out, proj_features

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, proj_features = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features

class VNet_2b(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_2b, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_five_up2 = Upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_six2 = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up2 = Upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_seven2 = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up2 = Upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)
        self.block_eight2 = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up2 = Upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.block_nine2 = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout2 = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def decoder2(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up2(x5)
        x5_up = x5_up + x4

        x6 = self.block_six2(x5_up)
        x6_up = self.block_six_up2(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven2(x6_up)
        x7_up = self.block_seven_up2(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight2(x7_up)
        x8_up = self.block_eight_up2(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine2(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout2(x9)
        out = self.out_conv2(x9)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        out2 = self.decoder2(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, out2
class VNet_multi_task(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, n_rotations = 4, layer_id=4, normalization='none', has_dropout=False):
        super(VNet_multi_task, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # self.linear1 = nn.Sequential(nn.Linear(n_filters * (1+2+4+8+16), n_filters),
        #                               nn.BatchNorm1d(n_filters),
        #                               nn.ReLU(inplace=True))
        #
        # self.linear2 = nn.Linear( n_filters, n_rotations)
        self.layer_id=layer_id
        factor = [1,2,4,8,16]
        self.linear1 = nn.Sequential(nn.Linear(n_filters * factor[self.layer_id], n_filters),
                                     nn.BatchNorm1d(n_filters),
                                     nn.ReLU(inplace=True))

        self.linear2 = nn.Linear(n_filters, n_rotations)

        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    # def rotation_predictor(self, features):
    #     x1 = features[0]
    #     x2 = features[1]
    #     x3 = features[2]
    #     x4 = features[3]
    #     x5 = features[4]
    #     f1 = x1.mean([2, 3, 4])
    #     f2 = x2.mean([2, 3, 4])
    #     f3 = x3.mean([2, 3, 4])
    #     f4 = x4.mean([2, 3, 4])
    #     f5 = x5.mean([2, 3, 4])
    #     feature = torch.cat([f1, f2, f3, f4, f5], 1)
    #     l1 = self.linear1(feature)
    #     l2 = self.linear2(l1)
    #     return l2
    def rotation_predictor(self, features):
        x5 = features[self.layer_id]
        f5 = x5.mean([2, 3, 4])
        l1 = self.linear1(f5)
        l2 = self.linear2(l1)
        return l2




    def forward(self, input, input_self_sup, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        if input_self_sup is not None:
            input_all = torch.cat([input, input_self_sup], 0)
            num_seg = input.size()[0]
            features = self.encoder(input_all)
            features_for_seg = []
            features_for_rotation_prediction = []
            for feature in features:
                features_for_seg.append(feature[:num_seg])
                features_for_rotation_prediction.append(feature[num_seg:])
            out = self.decoder(features_for_seg)
            out_rotation = self.rotation_predictor(features_for_rotation_prediction)
        else:
            features = self.encoder(input)
            out = self.decoder(features)
            out_rotation = None



        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, out_rotation

class VNet_dpb(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropblock=False, dropblock_rate=0.5, dropblock_size=3,):
        super(VNet_dpb, self).__init__()
        self.has_dropblock = has_dropblock

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        # self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.drop_block = DropBlock3D(block_size=dropblock_size, drop_prob=dropblock_rate)

        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)

        # if self.has_dropblock:
        #     x5 = self.drop_block(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropblock:
            x9 = self.drop_block(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False, noise_weight=None,uniform_range=None, num_feature_perturbated=None):
        if turnoff_drop:
            has_dropblock = self.has_dropblock
            self.has_dropblock = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(features[layer_id].device).unsqueeze(0) # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropblock = has_dropblock
        return out

class VNet_sdf(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet_sdf, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        out_tanh = self.tanh(out)
        out_seg = self.out_conv2(x9)
        return out_tanh, out_seg


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_tanh, out_seg = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out_tanh, out_seg

class VNetSupConDenseCL(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,head_layer_num=3, head_normalization='none'):
        super(VNetSupConDenseCL, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.head_enc = HeadConvBlock(head_layer_num, n_filters * 16, n_filters * 16, normalization=head_normalization)

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.head = HeadConvBlock(head_layer_num, n_classes, n_classes, normalization=head_normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        proj_enc_features_unnorm = self.head_enc(x5)
        proj_enc_features = F.normalize(proj_enc_features_unnorm, dim=1)

        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res, proj_enc_features

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        proj_features_unnorm = self.head(out)
        proj_features = F.normalize(proj_features_unnorm, dim=1)
        return out, proj_features

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features, proj_features_enc = self.encoder(input)

        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, proj_features = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features, proj_features_enc

class VNetSupConDenseCLAdaptive(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,head_layer_num=3, head_normalization='none',
                 denseCL_after_encoder=True, denseCLPos = 5, denseCL_Head_num = 3):
        super(VNetSupConDenseCLAdaptive, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        # self.head_enc = HeadConvBlock(head_layer_num, n_filters * 16, n_filters * 16, normalization=head_normalization)

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.head = HeadConvBlock(head_layer_num, n_classes, n_classes, normalization=head_normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.denseCL_after_encoder = denseCL_after_encoder
        if denseCL_after_encoder:
            if denseCLPos != 5:
                dense_filter = n_filters * (2 ** denseCLPos)

            else:
                dense_filter = n_filters * 16
        else:
            assert denseCLPos <= 8
            dense_filter = n_filters * (2 ** (8-denseCLPos))
        self.denseCL_head = HeadConvBlock(denseCL_Head_num, dense_filter, dense_filter, normalization=head_normalization)
        self.denseCLPos = denseCLPos
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        x5_before_dropout = x5

        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        if self.denseCL_after_encoder:
            if self.denseCLPos != 5:
                proj_dense_unnorm = self.denseCL_head(res[self.denseCLPos-1])
                proj_dense_features = F.normalize(proj_dense_unnorm, dim=1)
            else:
                proj_dense_unnorm = self.denseCL_head(x5_before_dropout)
                proj_dense_features = F.normalize(proj_dense_unnorm, dim=1)

        if self.denseCL_after_encoder:
            return res, proj_dense_features
        else:
            return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        proj_features_unnorm = self.head(out)
        proj_features = F.normalize(proj_features_unnorm, dim=1)

        res = [x5_up, x6_up, x7_up, x8_up]
        if not self.denseCL_after_encoder:
            proj_dense_unnorm = self.denseCL_head(res[self.denseCLPos - 5])
            proj_dense_features = F.normalize(proj_dense_unnorm, dim=1)

        if self.denseCL_after_encoder:
            return out, proj_features
        else:
            return out, proj_features, proj_dense_features

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        if self.denseCL_after_encoder:
            features, proj_dense_features = self.encoder(input)
        else:
            features = self.encoder(input)


        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        if self.denseCL_after_encoder:
            out, proj_features = self.decoder(features)
        else:
            out, proj_features, proj_dense_features = self.decoder(features)


        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features, proj_dense_features

class VNet_with_feature(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super(VNet_with_feature, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        features_decoder = [x6, x7, x8, x9]
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out, features_decoder
    def get_output_size(self,patch_size):
        return patch_size



    def forward(self, input, turnoff_drop=False, noise_weight=None,uniform_range=None, num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(features[layer_id].device).unsqueeze(0) # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, features_decoder = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, features_decoder

class VNet_with_projected_feature(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 head_pos_id=0, head_layer_num=3, head_normalization='none'):
        super(VNet_with_projected_feature, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        # self.head = HeadConvBlock(head_layer_num, n_classes, n_classes, normalization=head_normalization)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.head_pos_id = head_pos_id
        head_filter_num = n_filters * (2 ** head_pos_id)
        self.head = HeadConvBlock(head_layer_num, head_filter_num, head_filter_num, normalization=head_normalization)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        dec_features_to_proj = [x9, x8, x7, x6, x5]
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)

        proj_features_unnorm = self.head(dec_features_to_proj[self.head_pos_id])
        proj_features = F.normalize(proj_features_unnorm, dim=1)

        return out, proj_features

    def get_output_size(self, patch_size):
        return patch_size

    def forward(self, input, turnoff_drop=False, noise_weight=None, uniform_range=None,
                num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        if noise_weight is not None:
            uni_dist = Uniform(-uniform_range, uniform_range)
            for layer_id in range(len(features) - num_feature_perturbated, len(features)):
                noise_vector = noise_weight * uni_dist.sample(features[layer_id].size()[1:]).to(
                    features[layer_id].device).unsqueeze(0)  # if noise_weight: no noise
                features[layer_id] = features[layer_id].mul(noise_vector) + features[layer_id]
        out, proj_features = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out, proj_features


class VNetSplit10(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, head_num=10, normalization='none', has_dropout=False):
        super(VNetSplit10, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.heads = []
        self.head_num = head_num
        # for _ in range(head_num):
        #     self.heads.append(nn.Conv3d(n_filters, n_classes, 1, padding=0))
        self.out_conv = nn.Conv3d(n_filters, n_classes * head_num, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.n_classes = n_classes
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        all_out = []
        for head_id in range(self.head_num):
            all_out.append(self.heads[head_id](x9))
        return all_out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        print(self.training)
        return out

class VNetMultiHeadSplit10(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, head_num=10, n_filters=16, normalization='none', has_dropout=False):
        super(VNetMultiHeadSplit10, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes * head_num, 1, padding=0)
        self.n_classes = n_classes
        self.head_num = head_num

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False, noise_weight=None,uniform_range=None, num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        size = input.size()
        if self.training:
            return out
        else:
            return out.view([size[0], self.n_classes, self.head_num, size[2], size[3], size[4]])[:,:,0]

class GroupedConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, groups, normalization='none'):
        super(GroupedConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1, groups=groups))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
class VNetMultiHeadSplit9(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, head_num=10, n_filters=16, normalization='none', has_dropout=False):
        super(VNetMultiHeadSplit9, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = GroupedConvBlock(1, n_filters, n_filters, groups = head_num, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes * head_num, 1, padding=0, groups = head_num)
        self.n_classes = n_classes
        self.head_num = head_num

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False, noise_weight=None,uniform_range=None, num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        size = input.size()
        if self.training:
            return out
        else:
            return out.view([size[0], self.head_num, self.n_classes, size[2], size[3], size[4]])[:,0]

class GroupedUpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, groups, stride=2, normalization='none'):
        super(GroupedUpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride, groups=groups))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride, groups=groups))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class VNetMultiHeadSplit8(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, head_num=10, n_filters=16, normalization='none', has_dropout=False):
        super(VNetMultiHeadSplit8, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = GroupedConvBlock(2, n_filters * 2, n_filters * 2, groups = head_num * 2, normalization=normalization)
        self.block_eight_up = GroupedUpsamplingDeconvBlock(n_filters * 2, n_filters, groups = head_num * 2, normalization=normalization)

        self.block_nine = GroupedConvBlock(1, n_filters, n_filters, groups = head_num, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes * head_num, 1, padding=0, groups = head_num)
        self.n_classes = n_classes
        self.head_num = head_num

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out


    def forward(self, input, turnoff_drop=False, noise_weight=None,uniform_range=None, num_feature_perturbated=None):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        size = input.size()
        if self.training:
            return out
        else:
            return out.view([size[0], self.head_num, self.n_classes, size[2], size[3], size[4]])[:,0]

"""no activation after the last conv layer """
"""following every annotation accounts"""
"""use a sequence of 1 × 1 convolution, batch normalization [26] and ReLU non-linearity followed by a final 1 × 1 convolution"""
class HeadConvBlockDeepSup(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(HeadConvBlockDeepSup, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 1, padding=0))
            if i < n_stages - 1:
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_out))
                elif normalization == 'groupnorm':
                    ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
                elif normalization == 'instancenorm':
                    ops.append(nn.InstanceNorm3d(n_filters_out))
                elif normalization != 'none':
                    assert False
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
"""contract filter number at the last step"""
class HeadConvBlockDeepSup2(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(HeadConvBlockDeepSup2, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==n_stages - 1:
                output_channel = n_filters_out
            else:
                output_channel = n_filters_in

            ops.append(nn.Conv3d(n_filters_in, output_channel, 1, padding=0))
            if i < n_stages - 1:
                if normalization == 'batchnorm':
                    ops.append(nn.BatchNorm3d(n_filters_in))
                elif normalization == 'groupnorm':
                    ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_in))
                elif normalization == 'instancenorm':
                    ops.append(nn.InstanceNorm3d(n_filters_in))
                elif normalization != 'none':
                    assert False
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class VNet_ds_after8(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', head_type=1, has_dropout=False):
        super(VNet_ds_after8, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        if head_type == 1:
            self.block_eight_out = HeadConvBlockDeepSup(2, n_filters * 2, n_classes, normalization=normalization)
        else:
            self.block_eight_out = HeadConvBlockDeepSup2(2, n_filters * 2, n_classes, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        out_at8 = self.block_eight_out(x8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out_at8, out

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_at8, out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if self.training:
            return out_at8, out
        else:
            return out
class VNet_ds_after7(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', head_type=1, has_dropout=False):
        super(VNet_ds_after7, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        if head_type == 1:
            self.block_seven_out = HeadConvBlockDeepSup(2, n_filters * 4, n_classes, normalization=normalization)
        else:
            self.block_seven_out = HeadConvBlockDeepSup2(2, n_filters * 4, n_classes, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        if head_type == 1:
            self.block_eight_out = HeadConvBlockDeepSup(2, n_filters * 2, n_classes, normalization=normalization)
        else:
            self.block_eight_out = HeadConvBlockDeepSup2(2, n_filters * 2, n_classes, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        out_at7 = self.block_seven_out(x7)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        out_at8 = self.block_eight_out(x8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out_at7, out_at8, out

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_at7, out_at8, out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if self.training:
            return out_at7,out_at8, out
        else:
            return out
class VNet_ds_after6(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', head_type=1, has_dropout=False):
        super(VNet_ds_after6, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        if head_type == 1:
            self.block_six_out = HeadConvBlockDeepSup(2, n_filters * 8, n_classes, normalization=normalization)
        else:
            self.block_six_out = HeadConvBlockDeepSup2(2, n_filters * 8, n_classes, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        if head_type == 1:
            self.block_seven_out = HeadConvBlockDeepSup(2, n_filters * 4, n_classes, normalization=normalization)
        else:
            self.block_seven_out = HeadConvBlockDeepSup2(2, n_filters * 4, n_classes, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        if head_type == 1:
            self.block_eight_out = HeadConvBlockDeepSup(2, n_filters * 2, n_classes, normalization=normalization)
        else:
            self.block_eight_out = HeadConvBlockDeepSup2(2, n_filters * 2, n_classes, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        out_at6 = self.block_six_out(x6)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        out_at7 = self.block_seven_out(x7)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        out_at8 = self.block_eight_out(x8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out_at6, out_at7, out_at8, out

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_at6, out_at7, out_at8, out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if self.training:
            return out_at6, out_at7,out_at8, out
        else:
            return out

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww; out[0,i,j]=coord_flat[0,i] - coord_flat[0,j]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale # B_, num_heads, N, c//num_head; k.transpose(-2, -1): B_, num_heads, c//num_head, N
        attn = (q @ k.transpose(-2, -1)) # B_, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttention3D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww or Wd
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1) * (2 * self.window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wd-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords_d = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d]))  # 3, Wh, Ww, Wd
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd pair-wise distance
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 2] += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1) * (2 * self.window_size - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wd, Wh*Ww*Wd
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows*B, Wh*Ww*Wd, Wh*Ww*Wd) or None (originally mask (num_windows*B, Wh*Ww*Wd, Wh*Ww*Wd))
        """
        B_, N, C = x.shape # window_number, inner_window_pixel_number, channel
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B_, num_heads, N, c//num_head
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale # B_, num_heads, N, c//num_head; k.transpose(-2, -1): B_, num_heads, c//num_head, N
        attn = (q @ k.transpose(-2, -1)) # B_, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size * self.window_size, self.window_size * self.window_size * self.window_size, -1)  # Wh*Ww*Wd,Wh*Ww*Wd,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wd, Wh*Ww*Wd
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # attn = attn.view(-1, self.num_heads, N, N)
            # todo: sample-wise masking instead of positional masking
            attn = attn + mask.unsqueeze(1) #B_,self.num_heads, N, N
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

def window_partition3D(x, window_size):
    """
    Args:
        x: (B, H, W, D, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, D, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows

def mask_from_window_partition3D(x, window_size):
    """
    Args:
        x: (B, H, W, D)
        window_size (int): window size

    Returns:
        windows: (num_windows, Wh * Ww * Wd, Wh * Ww * Wd)
    """
    B, H, W, D = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6)
    pixel_num_window  = window_size * window_size * window_size
    # mask_conf = -float("Inf") * ((windows.sum(dim=[4,5,6]) == 0) + (windows.sum(dim=[4,5,6]) == pixel_num_window)).view(-1).float()# B, H // window_size, W // window_size, D // window_size,
    mask_windows = torch.zeros(B, H // window_size, W // window_size, D // window_size)
    mask_cond = ((windows.sum(dim=[4,5,6]) == 0) + (windows.sum(dim=[4,5,6]) == pixel_num_window))
    mask_windows[mask_cond] = -100
    mask_windows = mask_windows.view(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, pixel_num_window, pixel_num_window)
    return mask_windows.cuda()

def mask_01_from_window_partition3D(x, window_size):
    """
    Args:
        x: (B, H, W, D)
        window_size (int): window size

    Returns:
        windows: (num_windows, Wh * Ww * Wd)
    """
    B, H, W, D = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6)
    pixel_num_window  = window_size * window_size * window_size
    # mask_conf = -float("Inf") * ((windows.sum(dim=[4,5,6]) == 0) + (windows.sum(dim=[4,5,6]) == pixel_num_window)).view(-1).float()# B, H // window_size, W // window_size, D // window_size,
    mask_windows = torch.ones(B, H // window_size, W // window_size, D // window_size)
    mask_cond = ((windows.sum(dim=[4,5,6]) == 0) + (windows.sum(dim=[4,5,6]) == pixel_num_window))
    mask_windows[mask_cond] = 0
    mask_windows = mask_windows.view(-1).unsqueeze(-1).repeat(1, pixel_num_window)
    return mask_windows.cuda()

from skimage import morphology
def mask_01_from_dilated_window_partition3D(x, window_size):
    """
    Args:
        x: (B, H, W, D)
        window_size (int): window size

    Returns:
        windows: (num_windows, Wh * Ww * Wd)
    """
    B, H, W, D = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6)
    pixel_num_window  = window_size * window_size * window_size
    # mask_conf = -float("Inf") * ((windows.sum(dim=[4,5,6]) == 0) + (windows.sum(dim=[4,5,6]) == pixel_num_window)).view(-1).float()# B, H // window_size, W // window_size, D // window_size,
    mask_windows = torch.ones(B, H // window_size, W // window_size, D // window_size)
    mask_cond = ((windows.sum(dim=[4,5,6]) == 0) + (windows.sum(dim=[4,5,6]) == pixel_num_window)) # B, nWh, nWw, nWd
    mask_cond_boundary = ~mask_cond
    for b in range(B):
        mask_cond_boundary[b] = torch.from_numpy(morphology.binary_dilation(mask_cond_boundary[b].cpu(), morphology.ball(radius=1))).cuda()
    mask_cond = ~mask_cond_boundary
    mask_windows[mask_cond] = 0
    mask_windows = mask_windows.view(-1).unsqueeze(-1).repeat(1, pixel_num_window)
    return mask_windows.cuda()

def window_reverse3D(windows, window_size, H, W, D):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        D (int): Depth of image

    Returns:
        x: (B, H, W, D, C)
    """
    B = int(windows.shape[0] / (H * W * D / window_size / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
    return x

class VNet_dsba_after8(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', head_type=1, window_size = 2, self_atten_head_num = 1, sparse_attn = False, dilated_windows = False, has_dropout=False):
        super(VNet_dsba_after8, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        if head_type == 1: # todo: filter size
            self.block_eight_out = HeadConvBlockDeepSup(2, n_filters * 2, n_classes, normalization=normalization)
        else:
            self.block_eight_out = HeadConvBlockDeepSup2(2, n_filters * 2, n_classes, normalization=normalization)
        self.window_size = window_size
        self.sparse_attn = sparse_attn
        self.dilated_windows = dilated_windows
        self.window_eight = WindowAttention3D(n_filters * 2, window_size, num_heads=self_atten_head_num)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)# todo: filter size

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization) # todo: filter size
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder_before8(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        out_at8 = self.block_eight_out(x8)
        return x1, x8, out_at8

    def attention_at8(self, x8, pseudo_labels):
        # todo: partition input into windows
        B, C, D, H, W = x8.size()
        if H % self.window_size != 0 or W % self.window_size != 0 or D % self.window_size:
            padding = True
            H_ = (H // self.window_size + 1) * self.window_size
            W_ = (W // self.window_size + 1) * self.window_size
            D_ = (D // self.window_size + 1) * self.window_size
            # x8 = x8.view(-1, H, W, D)
            padding_op = nn.ReplicationPad3d((0, W_ - W, 0, H_ - H, 0, D_ - D ))
            x8 = padding_op(x8)# d,h,w
            pseudo_labels = padding_op(pseudo_labels.float().unsqueeze(1)).squeeze(1).long()
        else:
            padding = False

        x8_windows = window_partition3D(x8.permute(0,2,3,4,1), self.window_size)  # nW*B, window_size, window_size, window_size, C
        x8_ = x8_windows.view(-1, self.window_size * self.window_size * self.window_size, self.window_eight.dim)  # nW*B, window_size*window_size*window_size, C
        # todo: partition pseudo labels into windows
        if not self.sparse_attn:
            # near_boundary_mask_eight = mask_from_window_partition3D(pseudo_labels, self.window_size) #  mask: (0/-inf) mask with shape of (num_windows, Wh*Ww*Wd, Wh*Ww*Wd) or None: near-boundary masks
            near_boundary_mask_eight = None
            x8_atten = self.window_eight(x8_, mask=near_boundary_mask_eight)
            # merge windows
            attn_windows = x8_atten.view(-1, self.window_size, self.window_size, self.window_size, C) # (B_, N, C) -> (B_, ws, ws, ws, C)
        else:
            if self.dilated_windows:
                x8_atten = self.window_eight(x8_)  # B_, N, C
                near_boundary_mask_eight = mask_01_from_dilated_window_partition3D(pseudo_labels, self.window_size)  # B_, N
                attn_windows = (x8_atten * near_boundary_mask_eight.unsqueeze(-1)).view(-1, self.window_size, self.window_size, self.window_size, C)  # (B_, N, C) -> (B_, ws, ws, ws, C)
            else:
                x8_atten = self.window_eight(x8_) # B_, N, C
                near_boundary_mask_eight = mask_01_from_window_partition3D(pseudo_labels, self.window_size)# B_, N
                attn_windows = (x8_atten * near_boundary_mask_eight.unsqueeze(-1)).view(-1, self.window_size, self.window_size, self.window_size, C) # (B_, N, C) -> (B_, ws, ws, ws, C)

        # x8_after = torch.cat([x8, x8_atten], dim = 1) # todo: concate or add?
        if padding:
            x8_atten_rev = window_reverse3D(attn_windows, self.window_size, D_, H_, W_)  # B*nW, ws, ws, ws, C -> B, H, W, D, C
            x8_after = x8 + x8_atten_rev.permute(0, 4, 1, 2, 3)
            x8_after = x8_after[:, :, :D, :H, :W]
        else:
            x8_atten_rev = window_reverse3D(attn_windows, self.window_size, D, H, W)  # B*nW, ws, ws, ws, C -> B, H, W, D, C
            x8_after = x8 + x8_atten_rev.permute(0, 4, 1, 2, 3)
        return x8_after

    def decoder_after8(self, x1, x8_after):
        x8_up = self.block_eight_up(x8_after)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def forward(self, input, place_holder1 = None, step=1, turnoff_drop=False):
        if step == 1:
            # if turnoff_drop:
            #     has_dropout = self.has_dropout
            #     self.has_dropout = False
            features = self.encoder(input)
            x1, x8, out_at8 = self.decoder_before8(features)
            return x1, x8, out_at8
        elif step == 2:
            pseudo_labels = place_holder1
            x8_after = self.attention_at8(input, pseudo_labels) # x8 = input
            return x8_after
        else:
            assert step == 3
            x1, x8_after = input, place_holder1
            out = self.decoder_after8(x1, x8_after)
            return out
        # if turnoff_drop:
        #     self.has_dropout = has_dropout
        # if self.training:
        #     return out_at8, out
        # else:
        #     return out

    # x1, x8, out_at8 = model(image_in, step=1)
    # # todo: generate pseudo labels
    # # todo: + deep supervision loss
    # x8_after = model(x8, pseudo_labels = pseudo_labels, step=2)
    # out = model([x1,x8_after], step=3)
class VNet_dsba_after8_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', head_type=1, window_size = 2, self_atten_head_num = 1, sparse_attn = False,has_dropout=False):
        super(VNet_dsba_after8_v2, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        if head_type == 1: # todo: filter size
            self.block_eight_out = HeadConvBlockDeepSup(2, n_filters * 2, n_classes, normalization=normalization)
        else:
            self.block_eight_out = HeadConvBlockDeepSup2(2, n_filters * 2, n_classes, normalization=normalization)
        self.window_size = window_size
        self.sparse_attn = sparse_attn
        self.window_eight = WindowAttention3D(n_filters * 2, window_size, num_heads=self_atten_head_num)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)# todo: filter size

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization) # todo: filter size
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder_before8(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        out_at8 = self.block_eight_out(x8)
        return x1, x8, out_at8

    def attention_at8(self, x8, pseudo_labels, dilated_windows):
        # todo: partition input into windows
        B, C, D, H, W = x8.size()
        if H % self.window_size != 0 or W % self.window_size != 0 or D % self.window_size:
            padding = True
            H_ = (H // self.window_size + 1) * self.window_size
            W_ = (W // self.window_size + 1) * self.window_size
            D_ = (D // self.window_size + 1) * self.window_size
            # x8 = x8.view(-1, H, W, D)
            padding_op = nn.ReplicationPad3d((0, W_ - W, 0, H_ - H, 0, D_ - D ))
            x8 = padding_op(x8)# d,h,w
            pseudo_labels = padding_op(pseudo_labels.float().unsqueeze(1)).squeeze(1).long()
        else:
            padding = False

        x8_windows = window_partition3D(x8.permute(0,2,3,4,1), self.window_size)  # nW*B, window_size, window_size, window_size, C
        x8_ = x8_windows.view(-1, self.window_size * self.window_size * self.window_size, self.window_eight.dim)  # nW*B, window_size*window_size*window_size, C
        # todo: partition pseudo labels into windows
        if not self.sparse_attn:
            # near_boundary_mask_eight = mask_from_window_partition3D(pseudo_labels, self.window_size) #  mask: (0/-inf) mask with shape of (num_windows, Wh*Ww*Wd, Wh*Ww*Wd) or None: near-boundary masks
            near_boundary_mask_eight = None
            x8_atten = self.window_eight(x8_, mask=near_boundary_mask_eight)
            # merge windows
            attn_windows = x8_atten.view(-1, self.window_size, self.window_size, self.window_size, C) # (B_, N, C) -> (B_, ws, ws, ws, C)
        else:
            if dilated_windows:
                x8_atten = self.window_eight(x8_)  # B_, N, C
                near_boundary_mask_eight = mask_01_from_dilated_window_partition3D(pseudo_labels, self.window_size)  # B_, N
                attn_windows = (x8_atten * near_boundary_mask_eight.unsqueeze(-1)).view(-1, self.window_size, self.window_size, self.window_size, C)  # (B_, N, C) -> (B_, ws, ws, ws, C)
            else:
                x8_atten = self.window_eight(x8_) # B_, N, C
                near_boundary_mask_eight = mask_01_from_window_partition3D(pseudo_labels, self.window_size)# B_, N
                attn_windows = (x8_atten * near_boundary_mask_eight.unsqueeze(-1)).view(-1, self.window_size, self.window_size, self.window_size, C) # (B_, N, C) -> (B_, ws, ws, ws, C)

        # x8_after = torch.cat([x8, x8_atten], dim = 1) # todo: concate or add?
        if padding:
            x8_atten_rev = window_reverse3D(attn_windows, self.window_size, D_, H_, W_)  # B*nW, ws, ws, ws, C -> B, H, W, D, C
            x8_after = x8 + x8_atten_rev.permute(0, 4, 1, 2, 3)
            x8_after = x8_after[:, :, :D, :H, :W]
        else:
            x8_atten_rev = window_reverse3D(attn_windows, self.window_size, D, H, W)  # B*nW, ws, ws, ws, C -> B, H, W, D, C
            x8_after = x8 + x8_atten_rev.permute(0, 4, 1, 2, 3)
        return x8_after

    def decoder_after8(self, x1, x8_after):
        x8_up = self.block_eight_up(x8_after)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def forward(self, input, place_holder1 = None, step=1, dilated_windows=False, turnoff_drop=False):
        if step == 1:
            # if turnoff_drop:
            #     has_dropout = self.has_dropout
            #     self.has_dropout = False
            features = self.encoder(input)
            x1, x8, out_at8 = self.decoder_before8(features)
            return x1, x8, out_at8
        elif step == 2:
            pseudo_labels = place_holder1
            x8_after = self.attention_at8(input, pseudo_labels, dilated_windows) # x8 = input
            return x8_after
        else:
            assert step == 3
            x1, x8_after = input, place_holder1
            out = self.decoder_after8(x1, x8_after)
            return out
class VNet_dsba_after7(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', head_type=1, window_size = 2, self_atten_head_num = 1, has_dropout=False):
        super(VNet_dsba_after7, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        if head_type == 1: # todo: filter size
            self.block_eight_out = HeadConvBlockDeepSup(2, n_filters * 2, n_classes, normalization=normalization)
        else:
            self.block_eight_out = HeadConvBlockDeepSup2(2, n_filters * 2, n_classes, normalization=normalization)
        self.window_eight = WindowAttention3D(n_filters * 2, window_size, num_heads=self_atten_head_num)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)# todo: filter size

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization) # todo: filter size
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)


class VNet_dsba_after6(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', head_type=1, window_size = 2, self_atten_head_num = 1, has_dropout=False):
        super(VNet_dsba_after6, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        if head_type == 1: # todo: filter size
            self.block_eight_out = HeadConvBlockDeepSup(2, n_filters * 2, n_classes, normalization=normalization)
        else:
            self.block_eight_out = HeadConvBlockDeepSup2(2, n_filters * 2, n_classes, normalization=normalization)
        self.window_eight = WindowAttention3D(n_filters * 2, window_size, num_heads=self_atten_head_num)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)# todo: filter size

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization) # todo: filter size
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

