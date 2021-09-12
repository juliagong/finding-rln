import argparse
import logging
import os
from tqdm import tqdm
import pickle
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from torch.utils.tensorboard import SummaryWriter
from utils.dataset_retractor import BasicDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dice_loss import dice_coeff
from metrics import get_all_metrics

dir_img = '../data/images/'
dir_mask = '../data/annotations/'
dir_splits = '../data/'
dir_outputs = './outputs/'


def update_metrics(metrics, pred_bin, true_masks, convex, axis):
    for pred, true in zip(pred_bin, true_masks):
        curr = get_all_metrics(pred, true)
        for metric in metrics:
            metrics[metric] += curr[metric]
            if metric in ['convex_hull', 'rectangle_min']:
                print(f'{metric} dice: {curr[metric][1]}')
            if metric == 'convex_hull':
                convex.append(curr[metric][1])
            elif metric == 'rectangle_min':
                axis.append(curr[metric][1])


def add_tag_metrics(metrics_per_tag, img_name, img_tags, pred_bin, true_masks, dice):
    valid_tags = list(metrics_per_tag.keys())
    for tag in valid_tags:
        if tag in img_tags[img_name]:
            metrics_per_tag[tag].append(dice)
    metrics_per_tag[tuple(img_tags[img_name])].append(dice)


def test_net(net, loader, device, writer, tb_transform, global_step):
    net.eval()
    mask_type = torch.float32
    n_val = len(loader.dataset)  # the number of images
    bs = loader.batch_size
    scale = loader.dataset.scale
    tot = 0
    image_tags = pickle.load(open(os.path.join(dir_splits, 'image_tags.pkl'), 'rb'))
    valid_tags = list(np.unique(list(image_tags.values())))
    metrics_per_tag = {tag: [] for tag in valid_tags + list(itertools.product([t for t in valid_tags if 'lighting' in t], [t for t in valid_tags if 'zoom' in t]))}

    true_masks_stack, mask_pred_stack, img_stack = torch.zeros((n_val, int(1024*scale), int(1280*scale))), \
                                                   torch.zeros((n_val, 1, int(1024*scale), int(1280*scale))), \
                                                   torch.zeros((n_val, 3, int(1024*scale), int(1280*scale)))
    dices, convex, axis = [], [], []

    image_to_dice = {}
    with tqdm(total=n_val, desc='Test round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, true_masks, img_name, boxes = batch['image'], batch['mask'], batch['name'][0], batch['box']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type).sum(1)
            boxes = boxes.to(device=device, dtype=torch.float32)
            img_list = [img for img in imgs]

            with torch.no_grad():
                pred = net(img_list)
                if len(pred[0]['masks']) > 0:
                     accept_idxs = torch.tensor(np.argwhere(pred[0]['scores'].cpu() > 0.5).squeeze(0)).to(device)
                     accept_masks = torch.index_select(pred[0]['masks'], 0, accept_idxs)
                     final_mask = accept_masks.sum(0)  # all instances in one image
                     mask_pred = final_mask
                else:
                     mask_pred = torch.zeros_like(true_masks[0].unsqueeze(0))  # blank prediction

            pred_bin = (mask_pred > 0.5).float()
            curr_dice = dice_coeff(pred_bin, true_masks).item()  # average batch dice
            tot += curr_dice * true_masks.shape[0]  # add the total batch dice, not average
            dices.append(curr_dice)
            image_to_dice[img_name] = curr_dice
            add_tag_metrics(metrics_per_tag, img_name, image_tags, pred_bin.squeeze(1), true_masks, curr_dice)
            if bs == 1:  # log individual image dice scores
                writer.add_scalar('dice/test', curr_dice, global_step + i*bs)
                print(f'image {global_step + i*bs} dice: {curr_dice}')
            pbar.update()

            start, stop = i*bs, (i+1)*bs
            true_masks_stack[start:stop], mask_pred_stack[start:stop], img_stack[start:stop] = true_masks, pred_bin, imgs
# save images to disk; uncomment to save
#            save_image(pred_bin.squeeze(0), os.path.join(dir_outputs, 'f5', 'retractor', '{}.png'.format(i)))
#            save_image(true_masks.squeeze(0), os.path.join(dir_outputs, 'f5', 'true', '{}.png'.format(i)))
#            save_image(tb_transform(imgs.squeeze(0)), os.path.join(dir_outputs, 'f5', 'imgs', '{}.jpg'.format(i)))

        # log in tensorboard
        writer.add_image('images/test', make_grid(torch.stack([tb_transform(img) for img in img_stack]), padding=10, nrow=4), global_step)
        writer.add_image('masks/test/true', make_grid(true_masks_stack.unsqueeze(1), padding=10, nrow=4, pad_value=1), global_step)
        writer.add_image('masks/test/pred', make_grid(mask_pred_stack, padding=10, nrow=4, pad_value=1), global_step)
        writer.add_scalar('dice/test average', tot / n_val, global_step)


        # save individual image scores
        if os.path.exists(os.path.join(dir_splits, 'image_to_dice.pkl')):
            with open(os.path.join(dir_splits, 'image_to_dice.pkl'), 'rb') as f:
                image_to_dice_cum = pickle.load(f)
            for im in image_to_dice:
                image_to_dice_cum[im] = image_to_dice[im]  # overwrite if it exists
# uncomment to save per-image scores to disk
#            pickle.dump(image_to_dice_cum, open(os.path.join(dir_splits, 'image_to_dice.pkl'), 'wb'))
#        else:
#            pickle.dump(image_to_dice, open(os.path.join(dir_splits, 'image_to_dice.pkl'), 'wb'))

# uncomment to save per-tag scores to disk
#        if os.path.exists(os.path.join(dir_splits, 'metrics_per_tag.pkl')):
#            with open(os.path.join(dir_splits, 'metrics_per_tag.pkl'), 'rb') as f:
#                metrics_per_tag_cum = pickle.load(f)
#            for tag in metrics_per_tag:
#                metrics_per_tag_cum[tag] += metrics_per_tag[tag]
#            pickle.dump(metrics_per_tag_cum, open(os.path.join(dir_splits, 'metrics_per_tag.pkl'), 'wb'))
#        else:
#            pickle.dump(metrics_per_tag, open(os.path.join(dir_splits, 'metrics_per_tag.pkl'), 'wb'))
            
        for tag in metrics_per_tag:  # average metrics for each tag
            metrics_per_tag[tag] = np.mean(metrics_per_tag[tag])
            
    return tot / n_val, metrics_per_tag  # average test dice score, average per image tag


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
    net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = net.roi_heads.mask_predictor.conv5_mask.in_channels
    net.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info("Model loaded !")

    test = BasicDataset(dir_img, dir_mask, dir_splits, args.scale, train=False, val=False, test_split=args.test_split, augment=False) 
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    writer = SummaryWriter(comment=f'_TEST_SCALE_{args.scale}_fold{args.test_split}')
    global_step = 0
    tb_transform = transforms.Compose([transforms.Normalize([0., 0., 0.], [1/0.1951, 1/0.2208, 1/0.1968]),
                                       transforms.Normalize([-0.6641, -0.3883, -0.3489], [1., 1., 1.])])

    # test model and get average dice
    avg_dice, metrics_per_tag = test_net(net, test_loader, device, writer, tb_transform, global_step)
    print(f'average test dice: {avg_dice}')
    print(f'average test dice per setting: {metrics_per_tag}')
    writer.add_text('model_name', os.path.basename(args.model), 0)
    writer.close()
