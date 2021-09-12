import argparse
import logging
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch import optim
from tqdm import tqdm

from eval_retractor import eval_net
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.tensorboard import SummaryWriter
from utils.dataset_retractor import BasicDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from dice_loss import dice_coeff
from dice_loss import SoftJaccardBinary

dir_img = '../data/images/'
dir_mask = '../data/annotations/'
dir_splits = '../data/'
dir_logs = './logs/'


def get_sampler(patient_mapping, splits, contours, mode='train', test_split=5):
    assert mode in ['train', 'val'], 'samplers are used only for train and val sets'
    # image names and corresponding patient numbers
    trainval_splits = list([k for k in splits.keys() if k != test_split])
    if mode == 'val':
        val_split = (test_split-1 + 3) % 5 + 1
        imgs = [f for f in splits[val_split] if os.path.splitext(f)[0]+'.png' in contours]
    elif mode == 'train':
        imgs = [f for s in trainval_splits if s != test_split for f in splits[s] if os.path.splitext(f)[0]+'.png' in contours]

    num_samples = len(imgs)
    patients = [patient_mapping[img] for img in imgs]

    # counts for each patient
    patient_counts = {}  # map from patient ID to count of images with that ID
    for img in imgs:
        patient_counts[patient_mapping[img]] = patient_counts.get(patient_mapping[img], 0) + 1

    patient_weights = {patient : num_samples / patient_counts[patient] for patient in patient_counts}
    image_weights = [patient_weights[patients[i]] for i in range(num_samples)]
    sampler = WeightedRandomSampler(torch.DoubleTensor(image_weights), num_samples)
    return sampler


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=0.5,
              eval_every=1,
              augment=True,
              start_epoch=0,
              save_dir='checkpoints/',
              test_split=5):

    train = BasicDataset(dir_img, dir_mask, dir_splits, img_scale, train=True, val=False, test_split=test_split, augment=augment)
    val = BasicDataset(dir_img, dir_mask, dir_splits, img_scale, train=True, val=True, test_split=test_split, augment=False)
    n_train, n_val = len(train), len(val)
    splits = pickle.load(open(os.path.join(dir_splits, 'kfold_splits.pkl'), 'rb'))
    patient_mapping = pickle.load(open(os.path.join(dir_splits, 'patient_mapping.pkl'), 'rb'))
    contours = pickle.load(open(os.path.join(dir_splits, 'retractor_contours.pkl'), 'rb'))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    writer = SummaryWriter(comment=f'_TRAIN_LR_{lr}_BS_{batch_size}_SCALE_{img_scale}_f{test_split}')
    global_step = 0
    writer.add_text('dataset images/train', str([train.ids[i][:-1] for i in range(n_train)]))
    writer.add_text('dataset images/val', str([val.ids[i][:-1] for i in range(n_val)]))
    writer.add_text('train info/epochs', str(epochs))
    writer.add_text('train info/start epoch', str(start_epoch))
    writer.add_text('train info/augmentation', str(augment))
    writer.add_text('train info/image scale', str(img_scale))
    writer.add_text('train info/save directory', save_dir)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    def save_checkpoint(epoch, mode):
        try:
            os.mkdir(save_dir)
            logging.info('Created checkpoint directory')
        except OSError:
            pass
        torch.save(net.state_dict(),
                   os.path.join(save_dir, f'nerveidf{test_split}_{mode}_epoch{epoch + 1}.pth'))
        logging.info(f"{'Best ' + mode if mode != 'final' else 'Final'} checkpoint {epoch + 1} saved !")

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=50)

    tb_transform = transforms.Compose([transforms.Normalize([0., 0., 0.], [1/0.1951, 1/0.2208, 1/0.1968]),
                                       transforms.Normalize([-0.6641, -0.3883, -0.3489], [1., 1., 1.])])

    if (n_val + n_train) > 10 * batch_size:
        eval_every = (n_val + n_train) // (10 * batch_size)
    evals_per_epoch = len(train_loader) // eval_every

    best_dice, best_epoch, best_avg_dice, best_avg_epoch = 0, 0, 0, 0
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0
        epoch_dice = 0
        epoch_val_dice = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{start_epoch + epochs}', unit='img') as pbar:
            for batch in train_loader:
                net.train()
                imgs = batch['image']
                true_masks = batch['mask']
                boxes = batch['box']
                labels = batch['label']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)
                boxes = boxes.to(device=device, dtype=torch.float32)
                labels = labels.to(device)

                img_list = [img for img in imgs]
                tgt_list = [{'boxes': box, 'labels': label, 'masks': mask} for box, label, mask in zip(boxes, labels, true_masks)]

                loss_dict = net(img_list, tgt_list)
                loss = sum(loss for loss in loss_dict.values())
                epoch_loss += loss.item()
                writer.add_scalar('loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % eval_every == 0:
                    val_score, val_loss = eval_net(net, val_loader, device, writer, tb_transform, global_step)
                    epoch_val_dice += val_score
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('dice/val', val_score, global_step)
                    writer.add_scalar('loss/val', val_loss, global_step)

                    if val_score > best_dice:  # save the best epoch dice
                        best_dice = val_score
                        best_epoch = epoch
                        writer.add_scalar('dice/best_single', best_dice, global_step)
                        save_checkpoint(epoch, 'single')

            print('Epoch {}, avg loss {}, avg dice {}'.format(epoch, epoch_loss / len(train_loader), epoch_dice / len(train_loader)))
            scheduler.step(epoch_val_dice / evals_per_epoch)
            if epoch_val_dice / evals_per_epoch > best_avg_dice:  # save the best average dice
                best_avg_dice = epoch_val_dice / evals_per_epoch
                best_avg_epoch = epoch
                writer.add_scalar('dice/best_average', best_avg_dice, global_step)
                save_checkpoint(epoch, 'avg')
                
            print(f'Best single epoch so far: {best_epoch}, val dice {best_dice}')
            print(f'Best avg epoch so far: {best_avg_epoch}, val dice {best_avg_dice}')


    save_checkpoint(epoch, 'final')
    writer.add_text('best_epoch/single_batch', f'epoch {best_epoch}, val dice {best_dice}', global_step) 
    writer.add_text('best_epoch/average', f'epoch {best_avg_epoch}, val dice {best_avg_dice}', global_step) 
    print(f'Best epoch: {best_epoch}, val dice {best_dice}')
    print(f'Best average epoch: {best_avg_epoch}, val dice {best_avg_dice}')
    writer.close()


def init_weights(net, init_type='normal', init_gain=0.2):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-p', '--save_cp', dest='save_cp', type=bool, default=True,
                        help='Save checkpoints at all epochs for the model')
    parser.add_argument('-d', '--save_dir', dest='save_dir', type=str,
                        help='Directory in which to save checkpoints for the model')
    parser.add_argument('-t', '--test_split', type=int, default=5,
                        help='Which k-fold split is used for testing (1-5)', dest='test_split')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=100)
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # only one class
    in_features_mask = net.roi_heads.mask_predictor.conv5_mask.in_channels
    net.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)

    start_epoch = 0
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        start_epoch = int(args.load.split('/')[-1].split('.')[0].split('_')[-1].replace('epoch', ''))
        logging.info(f'Model loaded from {args.load}, starting from epoch {start_epoch + 1}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  save_cp=args.save_cp,
                  start_epoch=start_epoch,
                  save_dir=args.save_dir,
                  test_split=args.test_split)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupted model weights')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
