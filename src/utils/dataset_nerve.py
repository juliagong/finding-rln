from os.path import splitext, basename
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


class ImageMaskTransform:
    def __init__(self, augment, scale, pad_center, resize=True):
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
            iaa.Resize({"height": 1024, "width": 1280}) if resize else iaa.Resize(1.),
            iaa.Resize(scale),
        ])

    def __call__(self, img, mask, coords):
        img, mask, coords = np.array(img), np.array(mask), np.array(coords)
        mask = np.expand_dims(mask, [0,3])
        coords = np.expand_dims(coords, [0,1])
        if self.augment:
            img, mask, coords = self.aug(image=img, segmentation_maps=mask, bounding_boxes=coords)
        img, mask, coords = self.resize(image=img, segmentation_maps=mask, bounding_boxes=coords)
        mask = mask.squeeze(0)  # get rid of N dimension
        return img, mask, coords


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, splits_dir, scale=1., train=True, val=False, test_split=5, augment=True, use_pred_crops=True,
                 transform=transforms.Compose([transforms.Normalize([0.6641, 0.3883, 0.3489], [0.1951, 0.2208, 0.1968])])):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.train = train
        self.val = val
        self.test = not self.train and not self.val
        self.augment = augment
        self.augs = ImageMaskTransform(augment=self.augment, scale=self.scale, pad_center=(self.train and self.val or self.test))
        self.crop_augs = ImageMaskTransform(augment=False, scale=1., pad_center=(self.train and self.val or self.test), resize=False)
        self.transform = transform
        self.anatomy_index = 1 # png index value in masks to select for nerve
        self.use_pred_crops = use_pred_crops
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        assert (train and val) or (train and not val) or (not train and not val), 'Dataset cannot be val and test'

        # load image paths from correct split, add '.' to avoid prefix collision for glob
        splits = pickle.load(open(path.join(splits_dir, 'kfold_splits.pkl'), 'rb'))
        val_split = (test_split-1 + 3) % 5 + 1
        trainval_splits = list([k for k in splits.keys() if k != test_split])
        train_splits = [k for k in trainval_splits if k != val_split]
        if train and val:
            split = [splitext(f)[0] + '.' for f in splits[val_split]]
        elif train:  # all folds except for test fold
            split = [splitext(f)[0] + '.' for s in trainval_splits for f in splits[s]]
        else:  # test
            split = [splitext(f)[0] + '.' for f in splits[test_split]]
        self.ids = split

        # load ground truth crop coordinates for images that need to be cropped
        self.crops = pickle.load(open(path.join(splits_dir, 'pred_image_crops.pkl' if use_pred_crops else 'image_crops.pkl'), 'rb'))
        # truncate names to remove extension
        self.crops = {splitext(f)[0] + '.' : self.crops[f] for f in self.crops}

        logging.info(f'Created dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, is_rgb):
        img_nd = np.array(img)
        img_trans = img_nd.transpose((2, 0, 1))

        if is_rgb and img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans.astype(np.double) if is_rgb else img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        if img.mode != 'RGB':
            img = img.convert('RGB')  # all input images must have 3 channels
        if mask.mode != 'L':
            mask = mask.convert('L')    # all masks should be single-channel

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # if using ground-truth crops and image needs to be cropped, crop image
        if not self.use_pred_crops and idx in self.crops:
            coords = self.crops[idx]
            img = img.crop(coords)
            mask = mask.crop(coords)

        # filter mask to contain only pixels for nerve
        img = np.array(img)
        mask = np.array(mask)
        orig_h, orig_w = mask.shape

        mask[mask != self.anatomy_index] = 0
        mask[mask == self.anatomy_index] = 1
           
        # if vertical image, make horizontal
        if img.shape[0] > img.shape[1]:
            img = img.transpose((1,0,2))
            mask = mask.transpose((1,0))

        if self.use_pred_crops:
            coords = [0, 0, 0, 0]  # placeholder
        else:  # using ground-truth crops
            nerve_pixels = np.argwhere(mask == 1)
            ymin, ymax, xmin, xmax = nerve_pixels[:,0].min(), nerve_pixels[:,0].max(), nerve_pixels[:,1].min(), nerve_pixels[:,1].max()

        ia.seed(int((time.time()*1000)%100000))  # so that different batches get different augmentations
        img, mask, coords = self.crop_augs(img, mask, coords)

        if not self.use_pred_crops:  # pre-process if using ground-truth crops
            img, mask = self.preprocess(img, is_rgb=True), self.preprocess(mask, is_rgb=False)
            img = self.transform(torch.from_numpy(img).float())
            coords = torch.from_numpy(coords).float().flatten(0)

        elif self.use_pred_crops and idx in self.crops:  # if using predicted crops
            box_scale = max(orig_w, orig_h) / 320 if max(orig_w, orig_h) / min(orig_w, orig_h) > 1.25 else min(orig_w, orig_h) / 256
            x1, y1, x2, y2 = [round(c * box_scale) for c in self.crops[idx]]
            img, mask = img[y1:y2, x1:x2, :], mask[y1:y2, x1:x2].squeeze(2)
            nerve_pixels = np.argwhere(mask == 1)
            ymin, ymax, xmin, xmax = nerve_pixels[:,0].min(), nerve_pixels[:,0].max(), nerve_pixels[:,1].min(), nerve_pixels[:,1].max()
            ymin, ymax, xmin, xmax = max(0, ymin), min(mask.shape[0]-1, ymax), max(0, xmin), min(mask.shape[1]-1, xmax)
            coords = np.array([xmin, ymin, xmax, ymax])
            img, mask, coords = self.augs(img, mask, coords)
            img, mask = self.preprocess(img, is_rgb=True), self.preprocess(mask, is_rgb=False)
            img = self.transform(torch.from_numpy(img).float())
            coords = torch.from_numpy(coords).float().flatten(0)

        return {'image': img, 'mask': torch.from_numpy(mask), 'box': coords, 'name': basename(img_file[0])}
