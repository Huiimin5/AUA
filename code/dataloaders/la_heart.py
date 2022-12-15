import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler

class LAHeartMasked(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None, image_list_path='train.list'):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir + '/../' + image_list_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        mask = h5f['mask'][:]
        sample = {'image': image, 'label': label, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        return sample
class CenterCropMasked(object):
    def __init__(self, output_size, ):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            mask =  np.pad(mask, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        mask = mask[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label, 'mask': mask}


class RandomCropMasked(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, fixed=False):
        self.output_size = output_size
        self.fixed = fixed

    def __call__(self, sample):
        if self.fixed:
            np.random.seed(123)

        image, label, mask = sample['image'], sample['label'], sample['mask']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            mask = np.pad(mask, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)


        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        if self.fixed:
            w1, h1, d1 = 0, 0, 0

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        mask = mask[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        # if label.shape[-1] != 96:
        #     print('d:{}'.format(d))
        #     print('d1:{}'.format(d1))
        #     print('self.output_size[2]: {}'.format(self.output_size[2]))
        #     print(sample['image'].shape)
        #     print(sample['label'].shape)
        #     print(sample['mask'].shape)

        return {'image': image, 'label': label, 'mask': mask}


class RandomRotFlipMasked(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, fixed=False):
        self.fixed=fixed

    def __call__(self, sample):
        if self.fixed:
            np.random.seed(123)

        image, label, mask = sample['image'], sample['label'], sample['mask']
        k = np.random.randint(0, 4)
        if self.fixed:
            k = 1
        # todo: remove
        # k = 2
        # image = np.rot90(image, k).copy()
        # label = np.rot90(label, k).copy()
        # mask = np.rot90(mask, k).copy()
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        mask = np.rot90(mask, k)
        axis = np.random.randint(0, 2)
        if self.fixed:
            axis = 1
        # todo: remove
        # axis = 1
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()

        return {'image': image, 'label': label, 'mask': mask}

class RandomFlipMasked(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, fixed=False):
        self.fixed=fixed
    def __call__(self, sample):
        if self.fixed:
            np.random.seed(123)
        image, label, mask = sample['image'], sample['label'], sample['mask']
        axis = np.random.randint(0, 2)
        # todo: remove
        # axis = 1
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()

        return {'image': image, 'label': label, 'mask': mask}

class Random3DFlipMasked(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, fixed=False):
        self.fixed=fixed
    def __call__(self, sample):
        if self.fixed:
            np.random.seed(123)
        image, label, mask = sample['image'], sample['label'], sample['mask']
        axis = np.random.randint(0, 3)
        # todo: remove
        # axis = 1
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()

        return {'image': image, 'label': label, 'mask': mask}

import torchio
class GaussianBlurMasked(object):
    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        image = torchio.transforms.RandomBlur()(image[None, :])[0]
        mask = np.round(torchio.transforms.RandomBlur()(mask[None, :]))[0]
        return {'image': image, 'label': label, 'mask': mask}
class GaussianBlur(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torchio.transforms.RandomBlur()(image[None, :])[0]
        return {'image': image, 'label': label}

class RandomNoise(object):
    def __call__(self, sample, prob = 0.5):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        if np.random.choice((True, False), p=(prob, 1. - prob)):
            noise = np.clip(np.random.randn(*image.shape) * 0.1, -0.2, 0.2)
            image = image + noise
        return {'image': image, 'label': label, 'mask': mask}
class RandomAffineMasked(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, scales=(3/4, 4/3),fixed=False):
        self.scales = scales
        self.fixed=fixed

    def __call__(self, sample):
        if self.fixed:
            np.random.seed(123)

        image, label, mask = sample['image'], sample['label'], sample['mask']
        image = torchio.transforms.RandomAffine(scales=self.scales)(image[None, :])[0]
        mask = np.round(torchio.transforms.RandomAffine(scales=self.scales)(mask[None, :]))[0]
        return {'image': image, 'label': label, 'mask': mask}

class NoAugMasked(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, fixed=False):
        self.fixed=fixed

    def __call__(self, sample):

        image, label, mask = sample['image'], sample['label'], sample['mask']
        k = np.random.randint(0, 4)
        if self.fixed:
            k = 1
        # todo: remove
        k = 2
        # image = np.rot90(image, k).copy()
        # label = np.rot90(label, k).copy()
        # mask = np.rot90(mask, k).copy()
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        mask = np.rot90(mask, k)
        # new_image = np.rot90(image, k)
        # new_label = np.rot90(label, k)
        # new_mask = np.rot90(mask, k)
        axis = np.random.randint(0, 2)
        # todo: remove
        # axis = 1
        if self.fixed:
            axis = 1
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()

        # new_image = np.flip(image, axis=axis).copy()
        # new_label = np.flip(label, axis=axis).copy()
        # new_mask = np.flip(mask, axis=axis).copy()

        return {'image': image, 'label': label, 'mask': mask}

# class NoAugMasked(object):
#     """
#     Crop randomly flip the dataset in a sample
#     Args:
#     output_size (int): Desired output size
#     """
#     def __init__(self, fixed=False):
#         self.fixed=fixed
#
#     def __call__(self, sample):
#
#         image, label, mask = sample['image'], sample['label'], sample['mask']
#         k = np.random.randint(0, 4)
#         if self.fixed:
#             k = 1
#         # todo: remove
#         k = 2
#         # image = np.rot90(image, k).copy()
#         # label = np.rot90(label, k).copy()
#         # mask = np.rot90(mask, k).copy()
#         image = np.rot90(image, k)
#         label = np.rot90(label, k)
#         mask = np.rot90(mask, k)
#         # new_image = np.rot90(image, k)
#         # new_label = np.rot90(label, k)
#         # new_mask = np.rot90(mask, k)
#         axis = np.random.randint(0, 2)
#         # todo: remove
#         # axis = 1
#         if self.fixed:
#             axis = 1
#         image = np.flip(image, axis=axis).copy()
#         label = np.flip(label, axis=axis).copy()
#         mask = np.flip(mask, axis=axis).copy()
#
#         # new_image = np.flip(image, axis=axis).copy()
#         # new_label = np.flip(label, axis=axis).copy()
#         # new_mask = np.flip(mask, axis=axis).copy()
#
#         return {'image': image, 'label': label, 'mask': mask}

# class NoAugMasked(object):
#     """
#     Crop randomly flip the dataset in a sample
#     Args:
#     output_size (int): Desired output size
#     """
#     def __init__(self, fixed=False):
#         self.fixed=fixed
#
#
#     def __call__(self, sample):
#         if self.fixed:
#             np.random.seed(123)
#
#         image, label, mask = sample['image'], sample['label'], sample['mask']
#         k = np.random.randint(0, 4)
#         if self.fixed:
#             k = 1
#         # todo: remove
#         # k = 2
#         # image = np.rot90(image, k).copy()
#         # label = np.rot90(label, k).copy()
#         # mask = np.rot90(mask, k).copy()
#         image = np.rot90(image, k)
#         label = np.rot90(label, k)
#         mask = np.rot90(mask, k)
#         axis = np.random.randint(0, 2)
#         if self.fixed:
#             axis = 1
#         # todo: remove
#         # axis = 1
#         # image = np.flip(image, axis=axis).copy()
#         # label = np.flip(label, axis=axis).copy()
#         # mask = np.flip(mask, axis=axis).copy()
#         image = image.copy()
#         label = label.copy()
#         mask = mask.copy()
#
#         return {'image': image, 'label': label, 'mask': mask}
class RandomNoiseMasked(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label, 'mask': mask}


class CreateOnehotLabelMasked(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label, mask = sample['image'], sample['label'], sample['mask']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label, 'mask': mask}


class ToTensorMasked(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long(), 'mask': torch.from_numpy(sample['mask'])}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'mask': torch.from_numpy(sample['mask'])}

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/../train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size,fixed_order=False):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.fixed_order = fixed_order

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        if self.fixed_order:
            primary_iter = self.primary_indices
            secondary_iter = self.secondary_indices
            print(primary_iter)
            print(secondary_iter)
        else:
            primary_iter = iterate_once(self.primary_indices)
            secondary_iter = iterate_eternally(self.secondary_indices)

        res =  (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )
        # import pdb
        # pdb.set_trace()
        # for secondary_batch in grouper(secondary_iter, self.secondary_batch_size):
        #     print(secondary_batch)
        # if self.fixed_order:
        #     for _ in range(8):
        #         print(next(res))
        return res


    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class LabeledBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, batch_size,fixed_order=False):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size
        self.fixed_order = fixed_order

        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        if self.fixed_order:
            primary_iter = self.primary_indices
            print(primary_iter)
        else:
            primary_iter = iterate_once(self.primary_indices)

        res =  (
            primary_batch
            for primary_batch in grouper(primary_iter, self.primary_batch_size)
        )
        # import pdb
        # pdb.set_trace()
        # for secondary_batch in grouper(secondary_iter, self.secondary_batch_size):
        #     print(secondary_batch)
        # if self.fixed_order:
        #     for _ in range(8):
        #         print(next(res))
        return res


    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class UnlabeledBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, secondary_indices, secondary_batch_size,fixed_order=False):
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.fixed_order = fixed_order

        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        if self.fixed_order:
            secondary_iter = self.secondary_indices
            print(secondary_iter)
        else:
            secondary_iter = iterate_eternally(self.secondary_indices)

        res =  (secondary_batch
            for secondary_batch in grouper(secondary_iter, self.secondary_batch_size)
        )
        # import pdb
        # pdb.set_trace()
        # for secondary_batch in grouper(secondary_iter, self.secondary_batch_size):
        #     print(secondary_batch)
        # if self.fixed_order:
        #     for _ in range(8):
        #         print(next(res))
        return res

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

# augmentation
def transforms_for_rot(ema_inputs):

    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    # flip_mask = [0,0,0,0,1,1,1,1]
    # rot_mask = [0,1,2,3,0,1,2,3]

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_inputs[idx] = torch.flip(ema_inputs[idx], [1])

        ema_inputs[idx] = torch.rot90(ema_inputs[idx], int(rot_mask[idx]), dims=[1,2])

    return ema_inputs, rot_mask, flip_mask
def transforms_back_rot(ema_output,rot_mask, flip_mask):

    for idx in range(ema_output.shape[0]):

        ema_output[idx] = torch.rot90(ema_output[idx], int(rot_mask[idx]), dims=[2,1])

        if flip_mask[idx] == 1:
            ema_output[idx] = torch.flip(ema_output[idx], [1])

    return ema_output


if __name__ == "__main__":
    labeled_idxs = list(range(5))
    unlabeled_idxs = list(range(5, 100))
    batch_size = 4
    labeled_bs = 2
    from numpy import random
    random.seed(0)
    # sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    # for ids in sampler:
    #     print('ids {}'.format(ids))
        # print('id2 {}'.format(id2))

    labeled_sampler = LabeledBatchSampler(labeled_idxs, labeled_bs)
    unlabeled_sampler = UnlabeledBatchSampler(unlabeled_idxs, batch_size - labeled_bs)
    for epoch_num in range(25):
        for labeled_ids, unlabeled_ids in zip(labeled_sampler, unlabeled_sampler):
            ids = labeled_ids + unlabeled_ids
            print('ids {}'.format(ids))

    # labeled_sampler = LabeledBatchSampler(labeled_idxs, labeled_bs)
    # for labeled_ids in labeled_sampler:
    #     ids = labeled_ids
    #     print('ids {}'.format(ids))