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
import cv2


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
        mask = np.expand_dims(mask, 0)
        coords = np.expand_dims(coords, 0)
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
        self.anatomy_index = [15,19,20]  # png index values in masks to select for retractor
        self.max_retractors = 5
        self.use_pred_crops = use_pred_crops
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        assert (train and val) or (train and not val) or (not train and not val), 'Dataset cannot be val and test'

        # load ground truth retractor contours
        self.contours = pickle.load(open(path.join(splits_dir, 'retractor_contours.pkl'), 'rb'))

        # load image paths from correct split, add '.' to avoid prefix collision for glob
        splits = pickle.load(open(path.join(splits_dir, 'kfold_splits.pkl'), 'rb'))
        trainval_splits = list([k for k in splits.keys() if k != test_split])
        val_split = (test_split-1 + 3) % 5 + 1
        if train and val:
            split = [splitext(f)[2] + '.' for f in splits[val_split] if splitext(f)[0]+'.png' in self.contours]
        elif train:  # all folds except for test fold
            split = [splitext(f)[0] + '.' for s in trainval_splits if s != test_split for f in splits[s] if splitext(f)[0]+'.png' in self.contours]
        else:  # test
            split = [splitext(f)[0] + '.' for f in splits[test_split]]
        self.ids = split#[:8]

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

    def get_instance_masks(self, seg, cnt):
        # returns mask `ccs` (connected components) with one cc per channel
        ccs = []
        for c in cnt:
            # get connected component mask
            cc_mask = np.zeros_like(seg)
            rect = cv2.minAreaRect(c)
            box_r = cv2.boxPoints(rect)
            box_r = np.int0(box_r)
            cv2.drawContours(cc_mask,[box_r],-1,(1,1,1),-1)
            cc_mask = np.logical_and(cc_mask, seg).astype('uint8')
            ccs.append(cc_mask)
        return np.array(ccs)  # convert back to 0, 1 indexing

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

        # filter mask to contain only pixels for retractor
        img = np.array(img)
        mask = np.array(mask)
        mask[(mask != self.anatomy_index[0]) & (mask != self.anatomy_index[1]) & (mask != self.anatomy_index[2])] = 0
        mask[(mask == self.anatomy_index[0]) | (mask == self.anatomy_index[1]) | (mask == self.anatomy_index[2])] = 1
        orig_h, orig_w = mask.shape

        # get separate mask instances
        if idx + 'png' in self.contours:
            cnt = self.contours[idx + 'png']
            instance_masks = self.get_instance_masks(mask, cnt).transpose(1,2,0)  # H, W, num instances
        else:
            instance_masks = np.expand_dims(mask, 2)

        # if using ground-truth crops and image needs to be cropped, crop image
        if not self.use_pred_crops and idx in self.crops:
            x1, y1, x2, y2 = self.crops[idx]
            img, instance_masks = img[y1:y2, x1:x2, :], instance_masks[y1:y2, x1:x2, :]

        # if vertical image, make horizontal
        if img.shape[0] > img.shape[1]:
            img = img.transpose((1,0,2))
            instance_masks = instance_masks.transpose((1,0,2))

        if self.use_pred_crops:
            coords = np.zeros((instance_masks.shape[-1], 4))  # placeholder
        else:
            # TODO, not implemented; extrapolate from single-class case

        ia.seed(int((time.time()*1000)%100000))  # so that different batches get different augmentations
        img, instance_masks, coords = self.crop_augs(img, instance_masks, coords)

        if not self.use_pred_crops:  # pre-process if using ground-truth crops
            img, mask = self.preprocess(img, is_rgb=True), self.preprocess(mask, is_rgb=False)
            img = self.transform(torch.from_numpy(img).float())
            coords = torch.from_numpy(coords).float().flatten(0)

        elif self.use_pred_crops and idx in self.crops:  # if using predicted crops
            box_scale = max(orig_w, orig_h) / 320 if max(orig_w, orig_h) / min(orig_w, orig_h) > 1.25 else min(orig_w, orig_h) / 256
            x1, y1, x2, y2 = [round(c * box_scale) for c in self.crops[idx]]
            img, instance_masks = img[y1:y2, x1:x2, :], instance_masks[y1:y2, x1:x2, :]
            on_pixels = [np.argwhere(instance_masks[:,:,i] == 1) for i in range(instance_masks.shape[2])]  # on pixels for each instance
            coords = []
            for i in range(len(on_pixels)):
                if len(on_pixels[i]) > 0:
                    coords.append([on_pixels[i][:,1].min(), on_pixels[i][:,0].min(), on_pixels[i][:,1].max(), on_pixels[i][:,0].max()])
                else:
                    instance_masks = np.concatenate([instance_masks[:,:,:i], instance_masks[:,:,i+1:]], 2) if i+1 < instance_masks.shape[-1] else instance_masks[:,:,:i]
                    if instance_masks.shape[-1] == 0:
                        instance_masks = np.zeros((instance_masks.shape[0], instance_masks.shape[1], 1)).astype('uint8')
                        coords.append([0, 0, instance_masks.shape[1]-1, instance_masks.shape[0]-1])
            coords = np.array(coords)
            coords = np.array([[max(0, coords[i,0]), max(0, coords[i,1]), min(instance_masks.shape[1]-1, coords[i,2]), min(instance_masks.shape[0]-1, coords[i,3])] for i in range(len(coords))])
            img, instance_masks, coords = self.augs(img, instance_masks, coords)
            coords = coords.squeeze(0)
            img, instance_masks = self.preprocess(img, is_rgb=True), self.preprocess(instance_masks, is_rgb=False)
            labels = torch.ones(coords.shape[0]).to(torch.int64)
            if instance_masks.shape[0] < 5:
                instance_masks = np.concatenate([instance_masks, np.zeros((self.max_retractors - instance_masks.shape[0], instance_masks.shape[1], instance_masks.shape[2]))], 0).astype('uint8')
                coords = np.concatenate([coords, np.repeat(np.array([[0, 0, 319, 255]]), self.max_retractors - coords.shape[0], 0)], 0)
                labels = torch.cat([labels, torch.tensor([0 for i in range(self.max_retractors - len(labels))])], 0)
            img = self.transform(torch.from_numpy(img).float())
            coords = torch.from_numpy(coords).float()

        return {'image': img, 'mask': torch.from_numpy(instance_masks), 'box': coords, 'label': labels, 'name': basename(img_file[0])}
