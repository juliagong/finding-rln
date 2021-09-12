import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device, writer, tb_transform, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader.dataset)  # the number of images
    bs = loader.batch_size
    scale = loader.dataset.scale
    tot = 0
    loss = 0

    true_masks_stack, mask_pred_stack, img_stack = torch.zeros((n_val, int(1024*scale), int(1280*scale))), \
                                                   torch.zeros((n_val, int(1024*scale), int(1280*scale))), \
                                                   torch.zeros((n_val, 3, int(1024*scale), int(1280*scale)))
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, true_masks, boxes = batch['image'], batch['mask'], batch['box']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type).squeeze(1)
            boxes = boxes.to(device=device, dtype=torch.float32)
            img_list = [img for img in imgs]

            with torch.no_grad():
                pred = net(img_list)
                mask_pred = []
                for j in range(len(pred)):
                    if len(pred[j]['masks']) > 0:
                         mask_pred.append(pred[j]['masks'][0].squeeze(0))
                    else:
                         mask_pred.append(torch.zeros_like(true_masks[0]))  # blank prediction
                mask_pred = torch.stack(mask_pred)

            pred_bin = (mask_pred > 0.5).float()
            tot += dice_coeff(pred_bin, true_masks).item() * true_masks.shape[0]
            loss += F.binary_cross_entropy_with_logits(mask_pred.squeeze(1), true_masks).item() * true_masks.shape[0]
            pbar.update()

        # log in tensorboard: uncomment to log images
        #    start, stop = i*bs, min((i+1)*bs, i*bs + true_masks.shape[0])
        #    true_masks_stack[start:stop], mask_pred_stack[start:stop], img_stack[start:stop] = true_masks, mask_pred, imgs    
        #writer.add_image('images/val', make_grid(torch.stack([tb_transform(img) for img in img_stack]), padding=10, nrow=4), global_step)
        #writer.add_image('masks/val/true', make_grid(true_masks_stack.unsqueeze(1), padding=10, nrow=4), global_step)
        #writer.add_image('masks/val/pred', make_grid(torch.sigmoid(mask_pred_stack) > 0.5, padding=10, nrow=4), global_step)

    return tot / n_val, loss / n_val
