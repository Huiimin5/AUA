import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F
from .vnet import VNet, VNet_multi_task, VNet_dpb, VNet_2b, VNetSupCon, \
    VNet_noout, VNetSupCon_noout, VNetSupConDenseCL, VNetSupConDenseCLAdaptive,\
    VNetSupConMultiScale, VNetSupConMultiScaleFusion

import torch.distributions as td
import torch


from .base import ReshapedDistribution

# n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False):
class StochasticVNet(VNet):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits = F.relu(super().forward(image, noise_weight = noise_weight, uniform_range = uniform_range, num_feature_perturbated = num_feature_perturbated)) # [4, 16, 112, 112, 80]
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict
class StochasticVNet_noout(VNet_noout):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits = F.relu(super().forward(image, noise_weight = noise_weight, uniform_range = uniform_range, num_feature_perturbated = num_feature_perturbated)) # [4, 16, 112, 112, 80]
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict
class StochasticVNetSupCon(VNetSupCon):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 head_layer_num=3,
                 head_normalization='none',
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout,
                         head_layer_num,
                         head_normalization)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits_pre, features = super().forward(image )# [4, 16, 112, 112, 80]
        logits = F.relu( logits_pre)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, features, output_dict

class StochasticVNetSupConMultiScale(VNetSupConMultiScale):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 head_multi_scale_num=1,
                 head_layer_num=3,
                 head_normalization='none',
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout,
                         head_multi_scale_num,
                         head_layer_num,
                         head_normalization)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits_pre, features_multi_scale = super().forward(image )# [4, 16, 112, 112, 80]
        logits = F.relu( logits_pre)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, features_multi_scale, output_dict


class StochasticVNetSupConMultiScaleFusion(VNetSupConMultiScaleFusion):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 head_multi_scale_num=1,
                 head_layer_num=3,
                 head_normalization='none',
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout,
                         head_multi_scale_num,
                         head_layer_num,
                         head_normalization)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits_pre, features_multi_scale = super().forward(image )# [4, 16, 112, 112, 80]
        logits = F.relu( logits_pre)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, features_multi_scale, output_dict

class StochasticVNetSupCon_noout(VNetSupCon_noout):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 head_layer_num=3,
                 head_normalization='none',
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout,
                         head_layer_num,
                         head_normalization)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits_pre, features = super().forward(image )# [4, 16, 112, 112, 80]
        logits = F.relu( logits_pre)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, features,output_dict
class StochasticVNet_2b(VNet_2b):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

        self.mean_l2 = conv_fn(n_filters, num_classes, kernel_size=(1,) * self.dim)
        self.log_cov_diag_l2 = conv_fn(n_filters, num_classes, kernel_size=(1,) * self.dim)
        self.cov_factor_l2 = conv_fn(n_filters, num_classes * rank, kernel_size=(1,) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def from_vnet_to_output(self, logits, mean_l, log_cov_diag_l,cov_factor_l, mask, detach_cov ):
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = mean_l(logits)
        cov_diag = log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = cov_factor_l(logits)  # b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))  # b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3)  # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2)  # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view(
                (batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view(
                (batch_size, self.num_classes * self.rank) + event_shape[1:])
        return logit_mean, cov_diag_view, cov_factor_view, distribution


    def forward(self, image, mask,detach_cov = True):
        vnet_out_b1, vnet_out_b2 = super().forward(image)
        logits, logits2 = F.relu(vnet_out_b1),F.relu(vnet_out_b2) # [4, 16, 112, 112, 80]

        logit_mean, cov_diag_view, cov_factor_view, distribution = self.from_vnet_to_output(logits, self.mean_l, self.log_cov_diag_l, self.cov_factor_l, mask, detach_cov)

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        logit_mean2, cov_diag_view2, cov_factor_view2, distribution2 = self.from_vnet_to_output(logits2, self.mean_l2,self.log_cov_diag_l2,
                                                                                                self.cov_factor_l2, mask, detach_cov)

        output_dict2 = {'logit_mean': logit_mean2.detach(),
                       'cov_diag': cov_diag_view2,
                       'cov_factor': cov_factor_view2,
                       'distribution': distribution2}

        return logit_mean, output_dict, logit_mean2, output_dict2

class StochasticVNet_multi_task(VNet_multi_task):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_rotations=4,
                 n_filters=16,
                 layer_id=4,
                 normalization='none',
                 has_dropout=False,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         n_rotations,
                         layer_id,
                         normalization,
                         has_dropout)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def forward(self, image, input_self_sup, mask, detach_cov = True):
        logits_pre, rotation_logits = super().forward(image, input_self_sup) # [4, 16, 112, 112, 80]
        logits = F.relu(logits_pre)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict, rotation_logits

class StochasticVNet_dpb(VNet_dpb):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropblock=False,
                 dropblock_rate=0.5,
                 dropblock_size=3,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropblock,
                         dropblock_rate,
                         dropblock_size)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits = F.relu(super().forward(image, noise_weight = noise_weight, uniform_range = uniform_range, num_feature_perturbated = num_feature_perturbated)) # [4, 16, 112, 112, 80]
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict
from .unet import UNet

class StochasticUNet(UNet):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters)

        conv_fn = nn.Conv2d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 2
        # self.recomb_0 = conv_fn(n_filters, n_filters, kernel_size=(1, ) * self.dim)
        self.recomb_1 = conv_fn(n_filters, n_filters, kernel_size=(1,) * self.dim)
        self.recomb_2 = conv_fn(n_filters, n_filters, kernel_size=(1,) * self.dim)
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, detach_cov = True):
        pre_logits = F.relu(super().forward(image)) # [24, 16, 256, 256]
        # pre_logits = F.relu(self.recomb_0(pre_logits))
        pre_logits = F.relu(self.recomb_1(pre_logits))
        logits = F.relu(self.recomb_2(pre_logits))

        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w, r
        # cov_factor = cov_factor.view((batch_size, -1, self.rank))
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w,) -> (b, c, h, w,) -> b, c*h*w
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)
        output = distribution.rsample((1,))[0]

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return output, output_dict
from .unet import UNet_noout

class StochasticUNet_noout(UNet_noout):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters)

        conv_fn = nn.Conv2d # if self.dim == 3 else nn.Conv2d
        bn_fn = nn.BatchNorm2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 2
        self.recomb_0 = nn.Sequential(conv_fn(n_filters, n_filters, kernel_size=(3,) * self.dim, padding=1),bn_fn(n_filters))
        self.recomb_1 = nn.Sequential(conv_fn(n_filters, n_filters, kernel_size=(3,) * self.dim, padding=1),bn_fn(n_filters))
        self.recomb_2 = nn.Sequential(conv_fn(n_filters, n_filters, kernel_size=(3,) * self.dim, padding=1),bn_fn(n_filters))
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, detach_cov = True):
        pre_logits = F.relu(super().forward(image)) # [24, 16, 256, 256]
        pre_logits = F.relu(self.recomb_0(pre_logits))
        pre_logits = F.relu(self.recomb_1(pre_logits))
        logits = F.relu(self.recomb_2(pre_logits))

        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w, r
        # cov_factor = cov_factor.view((batch_size, -1, self.rank))
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w,) -> (b, c, h, w,) -> b, c*h*w
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)
        output = distribution.rsample((1,))[0]

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return output, output_dict

class StochasticVNetSupConDenseCL(VNetSupConDenseCL):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 head_layer_num=3,
                 head_normalization='none',
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout,
                         head_layer_num,
                         head_normalization)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits_pre, features, features_encoder = super().forward(image )# [4, 16, 112, 112, 80]
        logits = F.relu( logits_pre)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, features, features_encoder, output_dict

class StochasticVNetSupConDenseCLAdaptive(VNetSupConDenseCLAdaptive):
    def __init__(self,
                 input_channels,
                 num_classes,
                 n_filters=16,
                 normalization='none',
                 has_dropout=False,
                 head_layer_num=3,
                 head_normalization='none',
                 denseCL_after_encoder=True,
                 denseCLPos=5,
                 denseCL_Head_num=3,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(input_channels,
                         n_filters,
                         n_filters,
                         normalization,
                         has_dropout,
                         head_layer_num,
                         head_normalization,
                         denseCL_after_encoder,
                         denseCLPos , denseCL_Head_num)

        conv_fn = nn.Conv3d # if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        self.dim = 3
        self.mean_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(n_filters, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(n_filters, num_classes * rank, kernel_size=(1, ) * self.dim)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, mask, noise_weight=None, uniform_range = None, num_feature_perturbated = None, detach_cov = True):
        logits_pre, features, features_encoder = super().forward(image )# [4, 16, 112, 112, 80]
        logits = F.relu( logits_pre)
        batch_size = logits.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)# b,r*c, h, w, d
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))# b, r, c, h*w*d
        cov_factor = cov_factor.flatten(2, 3) # b, r, c*h*w*d
        cov_factor = cov_factor.transpose(1, 2) # b, c*h*w*d, r
        # import pdb
        # pdb.set_trace()

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        # (b, h, w, d) -> (b, c, h, w, d) -> b, c*h*w*d
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
        cov_factor = cov_factor * mask.unsqueeze(-1).float()
        cov_diag = cov_diag * mask.float() + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        if detach_cov:
            cov_diag_view = cov_diag.view(shape).detach()
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()
        else:
            cov_diag_view = cov_diag.view(shape)
            cov_factor_view = cov_factor.transpose(2, 1).contiguous().view((batch_size, self.num_classes * self.rank) + event_shape[1:])

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, features, features_encoder, output_dict