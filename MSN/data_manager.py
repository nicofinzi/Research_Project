# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
import subprocess
import time

from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import to_pil_image

from logging import getLogger

from PIL import ImageFilter
from PIL import Image

import torch
import torchvision

_GLOBAL_SEED = 0        # setting global seed for reproducibility
logger = getLogger()    # setting up logger object

def _labels_from_wrapper(ds):
    # works with PathMNISTWrapper before subsetting
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels).squeeze()
    if hasattr(ds, "dataset") and hasattr(ds.dataset, "labels"):
        return np.asarray(ds.dataset.labels).squeeze()
    raise RuntimeError("Could not find labels on dataset/wrapper for subsetting.")

def _balanced_indices(labels, per_class=100, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).squeeze()
    idxs = []
    for c in np.unique(labels):
        cls = np.where(labels == c)[0]
        take = min(per_class, len(cls))
        idxs.extend(rng.choice(cls, size=take, replace=False))
    rng.shuffle(idxs)
    return list(map(int, idxs))

def init_data(          # defining function to prepare DataLoader for dataset
    transform,          # data preprocesing and augmentation
    batch_size,         # no. of batches
    pin_mem=True,       # uses pin memory for faster GPU training
    num_workers=6,      # no. of CPU processes to use for data loading
    world_size=1,       # no. of distributed processes
    rank=0,             # ID of current process (0 = main process)
    root_path=None,     # base path
    image_folder=None,  # dataset name
    training=True,      # loading training, val or test split?
    copy_data=False,    # copies dataset locally before training
    drop_last=True,     # drop last batch due to incompleteness
    # NEW: choose explicit split when not training; and train label budget
    split=None,                         # "train" | "val" | "test"
    train_subset_frac=None,             # e.g., 0.1 for 10% labels
    train_subset_size=None,             # e.g., 9000
    train_subset_per_class=None,        # e.g., 100 per class
    train_subset_seed=42,               # reproducibility                        
):
    if image_folder == "pathmnist":   # creating instance of PathMNIST dataset
      # choose split
      split_name = split if split is not None else ('train' if training else 'test')
      dataset = PathMNISTWrapper(split=split_name, transform=transform)

      # train label-budget (few-label linear eval); only when training on labels
      if training and any(x is not None for x in (train_subset_frac, train_subset_size, train_subset_per_class)):
          labels = _labels_from_wrapper(dataset)
          N = len(labels)
          rng = np.random.default_rng(int(train_subset_seed))
          if train_subset_per_class is not None:
              idx = _balanced_indices(labels, per_class=int(train_subset_per_class), seed=int(train_subset_seed))
          elif train_subset_size is not None:
              k = max(1, min(int(train_subset_size), N))
              idx = list(rng.choice(np.arange(N), size=k, replace=False))
          else:  # frac
              k = max(1, int(round(float(train_subset_frac) * N)))
              idx = list(rng.choice(np.arange(N), size=k, replace=False))
          dataset = Subset(dataset, idx)
          logger.info(f"pathmnist TRAIN subset enabled: using {len(idx)} labeled samples "
                      f"(mode={'per-class' if train_subset_per_class is not None else 'size' if train_subset_size is not None else 'frac'})")

    # creates distributed sampler that ensures that each process gets a
    # different subset of the data
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,    # total no. of processes
        rank=rank)                  # rank (ID) of the current process
    # building DataLoader to feed data into the model
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=dist_sampler,       # handles data partitioning across GPUs
        batch_size=batch_size,      # no. of samples per batch
        drop_last=drop_last,        # decided whether to drop last batch
        pin_memory=pin_mem,         # speeds up data transfer to CUDA
        num_workers=num_workers,    # no. of subprocesses for loading data
        )
    # logs that DataLoader has been set up successfully
    logger.info('pathmnist unsupervised data loader created')

    # returns DataLoader and its associated distributed sampler
    return (data_loader, dist_sampler)


def make_transforms(
    rand_size=224,      # random views (global crops) - 224x224
    focal_size=96,      # focal views (local crops) - 96x96
    rand_crop_scale=(0.3, 1.0),     # global crop is 30%-100% of orignal area
    focal_crop_scale=(0.05, 0.3),   # local crop is 5%-30% of original area
    color_jitter=1.0,
    rand_views=2,       # no. of global views per sample
    focal_views=10,     # no. of focal views per sample
):
    # prints message to logger that the transforms are being built
    logger.info('making pathmnist data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        # applies jitter with 80% probability
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        # converts to grayscale with 20% probability
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    rand_transform = transforms.Compose([
        # randomly crops and resizes image
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        # flips the image horizontally with 50% probability
        transforms.RandomHorizontalFlip(),
        # applies color jitter and grayscale
        get_color_distortion(s=color_jitter),
        # blurs image with 50% probability
        GaussianBlur(p=0.5),
        # converts PIL image to PyTorch tensor
        transforms.ToTensor(),
        # applies mean and std normalisation
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    focal_transform = transforms.Compose([
        # randomly crops and resizes image
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        # flips the image horizontally with 50% probability
        transforms.RandomHorizontalFlip(),
        # applies color jitter and grayscale
        get_color_distortion(s=color_jitter),
        # blurs image with 50% probability
        GaussianBlur(p=0.5),
        # converts PIL image to PyTorch tensor
        transforms.ToTensor(),
        # applies mean and std normalisation
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # creates MultiViewTransform object that will
    # apply rand_transform for rand_views times
    # apply focal_transform for focal_views times
    transform = MultiViewTransform(
        rand_transform=rand_transform,
        focal_transform=focal_transform,
        rand_views=rand_views,
        focal_views=focal_views
    )
    return transform


# creates multiple augmented views (random and focal) of a single input image
class MultiViewTransform(object):

    def __init__(
        self,
        rand_transform=None,
        focal_transform=None,
        rand_views=1,         # no. of gloabl views to be generated per image
        focal_views=1,        # no. of local views to be generated per image
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []        # initialising empty list to store the
                              # augmented views

        # -- generate random views + adds them to the list
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- generate focal views + adds them to the list
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        # returns a list of all transformed views of the image
        return img_views


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p       # probability of applying the blur
        self.radius_min = radius_min    # min blur radius
        self.radius_max = radius_max    # max blur radius

    def __call__(self, img):
        # returns 1 with prob p, returns 0 with prob 1-p
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        # applies the blur using the randomly sampled radius above, with range
        # [radius_min, radius_max]
        return img.filter(ImageFilter.GaussianBlur(radius=radius.item()))


class PathMNISTWrapper(Dataset):
  def __init__(self, split='train', transform=None):
    self.dataset = PathMNIST(split=split, download=True)
    self.transform = transform

  def __len__(self):
    return len(self.dataset)    # returns no. of samples in pathmnist dataset

  def __getitem__(self, idx):
    img, label = self.dataset[idx]  # retrieves the image and label at index idx
    img = self.transform(img)
    # make label a scalar int
    y = int(label) if isinstance(label, (int, np.integer)) else int(label[0])
    return img, y
