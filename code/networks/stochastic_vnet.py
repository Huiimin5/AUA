import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F
from .vnet import VNetSupCon
import torch.distributions as td
import torch


from .base import ReshapedDistribution

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
