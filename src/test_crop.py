import argparse
import logging
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from torch.utils.tensorboard import SummaryWriter
from utils.dataset_crop_test import BasicDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dice_loss import dice_coeff
from metrics import get_all_metrics
from train_crop import box_iou

dir_img = '../data/images/'
dir_mask = '../data/annotations/'
dir_splits = '../data/'
dir_outputs = './outputs/'


def update_metrics(metrics, pred_bin, true_masks, ious):
    for pred, true in zip(pred_bin, true_masks):
        curr = get_all_metrics(pred, true)
        for metric in metrics:
            metrics[metric] += curr[metric]
            if metric in ['convex_hull', 'rectangle_min']:
                print(f'{metric} dice: {curr[metric][1]}')
            if metric == 'iou':
                ious.append(curr[metric].item() if curr[metric] != 0 else curr[metric])


def test_net(net, loader, device, writer, tb_transform, global_step):
    net.eval()
    n_val = len(loader.dataset)  # the number of images
    bs = loader.batch_size
    scale = loader.dataset.scale
    tot = 0
    metrics = {'convex_hull': np.zeros(2), 'rectangle_axis': np.zeros(2), 'rectangle_min': np.zeros(2),
               'circle': np.zeros(2), 'ellipse': np.zeros(2), 'slope': 0., 'iou': 0.}

    true_masks_stack, mask_pred_stack, img_stack = torch.zeros((n_val, int(1024*scale), int(1280*scale))), \
                                                   torch.zeros((n_val, 1, int(1024*scale), int(1280*scale))), \
                                                   torch.zeros((n_val, 3, int(1024*scale), int(1280*scale)))
    coord_stack, ious, dices = [], [], []

    with tqdm(total=n_val, desc='Test round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32).squeeze(1)

            with torch.no_grad():
                coords = net(imgs).squeeze(0)
                x1, y1, x2, y2 = max(0, coords[0].item()), max(0, coords[1].item()), max(0, coords[2].item()), max(0, coords[3].item())
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                coord_stack.append([x1, y1, x2, y2])
                mask_pred = torch.zeros_like(true_masks.unsqueeze(1))
                mask_pred[:, :, y1:y2, x1:x2] = 1

            pred = torch.sigmoid(mask_pred)
            pred_bin = (pred > 0.5).float()
            curr_dice = dice_coeff(pred_bin, true_masks).item()  # average batch dice
            tot += curr_dice * true_masks.shape[0]  # add the total batch dice, not average
            dices.append(curr_dice)
            update_metrics(metrics, pred_bin.squeeze(1), true_masks, ious)  # compute running metrics other than dice
            if bs == 1:  # log individual image dice scores
                writer.add_scalar('dice/test', curr_dice, global_step + i*bs)
                print(f'image {global_step + i*bs} dice: {curr_dice}')
            pbar.update()

            start, stop = i*bs, (i+1)*bs
            true_masks_stack[start:stop], mask_pred_stack[start:stop], img_stack[start:stop] = true_masks, pred_bin, imgs
# save images to disk; uncomment to save
#            save_image(pred_bin.squeeze(0), os.path.join(dir_outputs, 'retractor', 'pred', '{}.png'.format(i)))
#            save_image(true_masks.squeeze(0), os.path.join(dir_outputs, 'retractor', 'true', '{}.png'.format(i)))
#            save_image(tb_transform(imgs.squeeze(0)), os.path.join(dir_outputs, 'imgs', '{}.jpg'.format(i)))

        # log in tensorboard
        writer.add_image('images/test', make_grid(torch.stack([tb_transform(img) for img in img_stack]), padding=10, nrow=4), global_step)
        writer.add_image('masks/test/true', make_grid(true_masks_stack.unsqueeze(1), padding=10, nrow=4, pad_value=1), global_step)
        writer.add_image('masks/test/pred', make_grid(mask_pred_stack, padding=10, nrow=4, pad_value=1), global_step)
        writer.add_scalar('dice/test average', tot / n_val, global_step)
        writer.add_text('bounding_boxes', str(coord_stack), global_step)

        # compute average metrics
        for metric in metrics:
            metrics[metric] = metrics[metric] / n_val
            if metric not in ['slope', 'iou']:  # both iou and dice
                writer.add_scalar(f'dice/{metric} average', metrics[metric][0], global_step)
                writer.add_scalar(f'iou/{metric} average', metrics[metric][1], global_step)
            else:
                writer.add_scalar('{metric} average', metrics[metric], global_step)

    return tot / n_val, metrics  # average test dice score, average other metrics


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-r', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('-a', '--anatomy', dest='anatomy', type=str, default='nerve',
                        help='Anatomical feature to train on')
    parser.add_argument('-t', '--test_split', type=int, default=5,
                        help='Which k-fold split is used for testing (1-5)', dest='test_split')

    return parser.parse_args()


if __name__ == "__main__":
    # get arguments
    args = get_args()

    # load model
    net = torchvision.models.resnext50_32x4d(pretrained=True)
    net.fc = nn.Linear(in_features=2048, out_features=4)
    logging.info("Loading model {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    test = BasicDataset(dir_img, dir_mask, dir_splits, args.scale, train=False, val=False, test_split=args.test_split, augment=False) 
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    writer = SummaryWriter(comment=f'_TEST_SCALE_{args.scale}')
    global_step = 0
    tb_transform = transforms.Compose([transforms.Normalize([0., 0., 0.], [1/0.1951, 1/0.2208, 1/0.1968]),
                                       transforms.Normalize([-0.6641, -0.3883, -0.3489], [1., 1., 1.])])

    # test model and get average dice
    avg_dice, metrics = test_net(net, test_loader, device, writer, tb_transform, global_step)
    print(f'average test dice: {avg_dice}')    
    print('average test iou: {}'.format(metrics['iou']))
    writer.add_text('model_name', os.path.basename(args.model), 0)
    writer.close()
