from os.path import splitext
from os import listdir, path
import numpy as np
import pickle
from glob import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import logging
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import random
import time


class ImageTransform:
    def __init__(self, augment, scale, pad_center):
        self.augment = augment
        # augmentations
        self.aug = iaa.Sequential([
            iaa.SomeOf((2, 4),
            [   
                iaa.Affine(scale=(0.8,1.2)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                iaa.Rotate((-30, 30)),
                iaa.PerspectiveTransform(scale=(0.01, 0.05)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ]),
            iaa.SomeOf((1, 3),
            [
                iaa.MultiplyBrightness((0.6, 1.4)),
                iaa.MultiplySaturation((0.8, 1.2)),
                iaa.GammaContrast((0.5, 2.0))
            ])
        ])
        # pad and resize as needed to minimize image padding
        self.resize = iaa.Sequential([
            iaa.CenterPadToAspectRatio(1.25) if pad_center else iaa.PadToAspectRatio(1.25),
            iaa.Resize({"height": 1024, "width": 1280}),
            iaa.Resize(scale),
        ])

    def __call__(self, img, coords):
        img, coords = np.array(img), np.array(coords)
        coords = np.expand_dims(coords, [0,1])
        if self.augment:
            img, coords = self.aug(image=img, bounding_boxes=coords)
        img, coords = self.resize(image=img, bounding_boxes=coords)
        return img, coords


class CropDataset(Dataset):
    def __init__(self, imgs_dir, splits_dir, scale=1, train=True, val=False, test_split=5, augment=True,
                 transform=transforms.Compose([transforms.Normalize([0.6641, 0.3883, 0.3489], [0.1951, 0.2208, 0.1968])])):
        self.imgs_dir = imgs_dir
        self.scale = scale
        self.train = train
        self.val = val
        self.test = not self.train and not self.val
        self.augment = augment
        self.transform = transform
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        assert (train and val) or (train and not val) or (not train and not val), 'Dataset cannot be val and test'

        # load image paths from correct split, add '.' to avoid prefix collision for glob
        splits = pickle.load(open(path.join(splits_dir, 'kfold_splits.pkl'), 'rb'))
        trainval_splits = list([k for k in splits.keys() if k != test_split])
        val_split = (test_split-1 + 3) % 5 + 1
        if train and val:
            split = [splitext(f)[0] + '.' for f in splits[val_split]]
        elif train:  # all folds except for test fold
            split = [splitext(f)[0] + '.' for s in trainval_splits if s != test_split for f in splits[s]]
        else:  # test
            split = [splitext(f)[0] + '.' for f in splits[test_split]]
        self.ids = split

        # load ground truth crop coords
        self.crops = pickle.load(open(path.join(splits_dir, 'image_crops.pkl'), 'rb'))
        # truncate names to remove extension
        self.crops = {splitext(f)[0] + '.' : self.crops[f] for f in self.crops}

        logging.info(f'Created dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img):
        img_nd = np.array(img)
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans.astype(np.double)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        img = Image.open(img_file[0])
        img_dims = torch.tensor([img.size[0], img.size[1]])

        # if macro image, get ground truth crop; else, ground truth is whole image
        if idx in self.crops:
            coords = torch.tensor(self.crops[idx])
        else:
            coords = torch.tensor([0, 0, img.size[0] - 1, img.size[1] - 1])

        if img.mode != 'RGB':
            img = img.convert('RGB')  # all input images must have 3 channels

        # filter mask to contain only pixels for nerve
        img = np.array(img)
           
        # if vertical image, make horizontal
        if img.shape[0] > img.shape[1]:
            img = img.transpose((1,0,2))
            coords = [coords[1], coords[0], coords[3], coords[2]]  # rotate 90 degrees cc and flip vertical

        ia.seed(int((time.time()*1000)%100000))  # so that different batches get different augmentations
        augs = ImageTransform(augment=self.augment, scale=self.scale, pad_center=(self.train and self.val or self.test))
        img, coords = augs(img, coords)
        img = self.preprocess(img)
        img = self.transform(torch.from_numpy(img).float())
        coords = torch.from_numpy(coords).float().flatten(0)

        return img, {'boxes': coords, 'image_dims': img_dims}  # there is only one class
