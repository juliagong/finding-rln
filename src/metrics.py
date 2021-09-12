import cv2
import numpy as np
import torch
from dice_loss import dice_coeff


def get_contours(mask):
    """
    Returns the contour of the provided segmentation mask.
    """
    mask = mask * 255

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    _, cnt, _ = cv2.findContours(mask, 2, 1)
    if len(cnt) > 0:
        cnt = cnt[0]

    return cnt


def iou(pred, true):
    eps = 0.0001
    pred, true = pred.float(), true.float()
    inter = torch.dot(pred.view(-1), true.view(-1))
    union = torch.sum(pred) + torch.sum(true) + eps
    return (inter + eps) / (union - inter)


def dice(pred, true):
    return dice_coeff(pred.unsqueeze(0).float(), true.unsqueeze(0).float())


def convex_hull_metrics(true, pred, true_cnt, pred_cnt):
    """
    Returns the convex hull iou and dice scores of the segmentation.
    `true`, `pred` are the segmentation masks.
    `true_cnt`, `pred_cnt` are the contours of the masks.
    """
    true_hull = cv2.convexHull(true_cnt, returnPoints = True)
    pred_hull = cv2.convexHull(pred_cnt, returnPoints = True)
    
    true_mask = cv2.cvtColor(true.copy(), cv2.COLOR_GRAY2RGB)
    cv2.fillConvexPoly(true_mask, true_hull, [255,255,255])
    
    pred_mask = cv2.cvtColor(pred.copy(), cv2.COLOR_GRAY2RGB)
    cv2.fillConvexPoly(pred_mask, pred_hull, [255,255,255])
    
    pred_mask = torch.from_numpy(pred_mask[:,:,0])
    true_mask = torch.from_numpy(true_mask[:,:,0])

    pred_mask[pred_mask != 0] = 1.
    true_mask[true_mask != 0] = 1.
    
    return iou(pred_mask, true_mask), dice(pred_mask, true_mask)


def rect_metrics(true, pred, true_cnt, pred_cnt):
    """
    Returns the bounding box dice scores of the segmentations;
    first the axis-aligned, then the minimum-area iou and then dice respectively.
    
    `true`, `pred` are the segmentation masks.
    `true_cnt`, `pred_cnt` are the contours of the masks.
    """
    # axis-aligned bounding box
    true_mask = cv2.cvtColor(true.copy(), cv2.COLOR_GRAY2RGB)
    x,y,w,h = cv2.boundingRect(true_cnt)
    cv2.rectangle(true_mask,(x,y),(x+w,y+h),(255,255,255),-1)
    
    pred_mask = cv2.cvtColor(pred.copy(), cv2.COLOR_GRAY2RGB)
    x,y,w,h = cv2.boundingRect(pred_cnt)
    cv2.rectangle(pred_mask,(x,y),(x+w,y+h),(255,255,255),-1)
    
    pred_mask = torch.from_numpy(pred_mask[:,:,0])
    true_mask = torch.from_numpy(true_mask[:,:,0])
    pred_mask[pred_mask != 0] = 1.
    true_mask[true_mask != 0] = 1.
    axis_aligned_dice = dice(pred_mask, true_mask)
    axis_aligned_iou = iou(pred_mask, true_mask)
    
    
    # minimum-area rotated bounding box
    true_mask = cv2.cvtColor(true.copy(), cv2.COLOR_GRAY2RGB)
    rect = cv2.minAreaRect(true_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(true_mask,[box],-1,(255,255,255),-1)
    
    pred_mask = cv2.cvtColor(pred.copy(), cv2.COLOR_GRAY2RGB)
    rect = cv2.minAreaRect(pred_cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(pred_mask,[box],-1,(255,255,255),-1)
    
    pred_mask = torch.from_numpy(pred_mask[:,:,0])
    true_mask = torch.from_numpy(true_mask[:,:,0])
    pred_mask[pred_mask != 0] = 1.
    true_mask[true_mask != 0] = 1.
    min_area_dice = dice(pred_mask, true_mask)
    min_area_iou = iou(pred_mask, true_mask)
    
    return axis_aligned_iou, min_area_iou, axis_aligned_dice, min_area_dice


def ellipse_metrics(true, pred, true_cnt, pred_cnt):
    # mininum enclosing circle
    true_mask = cv2.cvtColor(true.copy(), cv2.COLOR_GRAY2RGB)
    (x,y), radius = cv2.minEnclosingCircle(true_cnt)
    center = (int(x), int(y))
    radius = int(radius)
    true_mask = np.zeros(true_mask.shape)
    cv2.circle(true_mask,center,radius,(255,255,255),-1)
    
    pred_mask = cv2.cvtColor(pred.copy(), cv2.COLOR_GRAY2RGB)
    (x,y), radius = cv2.minEnclosingCircle(pred_cnt)
    center = (int(x), int(y))
    radius = int(radius)
    pred_mask = np.zeros(pred_mask.shape)
    cv2.circle(pred_mask,center,radius,(255,255,255),-1)
    
    pred_mask = torch.from_numpy(pred_mask[:,:,0])
    true_mask = torch.from_numpy(true_mask[:,:,0])
    pred_mask[pred_mask != 0] = 1.
    true_mask[true_mask != 0] = 1.
    circle_dice = dice(pred_mask, true_mask)
    circle_iou = iou(pred_mask, true_mask)
    
    
    # fitted ellipse (not necessarily enclosing)
    if len(true_cnt) < 5 or len(pred_cnt) < 5:
        return circle_iou, 0, circle_dice, 0
    true_mask = cv2.cvtColor(true.copy(), cv2.COLOR_GRAY2RGB)
    ellipse = cv2.fitEllipse(true_cnt)
    true_mask = np.zeros(true_mask.shape)
    cv2.ellipse(true_mask,ellipse,(255,255,255),-1)
    
    pred_mask = cv2.cvtColor(pred.copy(), cv2.COLOR_GRAY2RGB)
    ellipse = cv2.fitEllipse(pred_cnt)
    pred_mask = np.zeros(pred_mask.shape)
    cv2.ellipse(pred_mask,ellipse,(255,255,255),-1)
    
    pred_mask = torch.from_numpy(pred_mask[:,:,0])
    true_mask = torch.from_numpy(true_mask[:,:,0])
    pred_mask[pred_mask != 0] = 1.
    true_mask[true_mask != 0] = 1.
    ellipse_dice = dice(pred_mask, true_mask)
    ellipse_iou = iou(pred_mask, true_mask)
    
    return circle_iou, ellipse_iou, circle_dice, ellipse_dice


def line_metric(true_cnt, pred_cnt):
    [vx,vy,x,y] = cv2.fitLine(true_cnt, cv2.DIST_L2,0,0.01,0.01)
    true_slope = vy/vx
    
    [vx,vy,x,y] = cv2.fitLine(pred_cnt, cv2.DIST_L2,0,0.01,0.01)
    pred_slope = vy/vx
    
    return abs(true_slope - pred_slope)[0]


############################ function wrapper to calculate metrics
def get_all_metrics(pred, true):
    pred, true = np.uint8(pred.cpu()), np.uint8(true.cpu())
    true_cnt = get_contours(true)
    pred_cnt = get_contours(pred)
    if len(true_cnt) == 0 or len(pred_cnt) == 0:
        return {'convex_hull': np.zeros(2), 'rectangle_axis': np.zeros(2), 'rectangle_min': np.zeros(2),
                'circle': np.zeros(2), 'ellipse': np.zeros(2), 'slope': 0., 'iou': 0.}

    convex_hull_iou, convex_hull_dice = convex_hull_metrics(true, pred, true_cnt, pred_cnt)
    rect_axis_iou, rect_min_iou, rect_axis_dice, rect_min_dice = rect_metrics(true, pred, true_cnt, pred_cnt)
    circle_iou, ellipse_iou, circle_dice, ellipse_dice = ellipse_metrics(true, pred, true_cnt, pred_cnt)
    slope_diff = line_metric(true_cnt, pred_cnt)

    return {'convex_hull': np.array([convex_hull_iou, convex_hull_dice]),
            'rectangle_axis': np.array([rect_axis_iou, rect_axis_dice]),
            'rectangle_min': np.array([rect_min_iou, rect_min_dice]),
            'circle': np.array([circle_iou, circle_dice]),
            'ellipse': np.array([ellipse_iou, ellipse_dice]),
            'slope': slope_diff,
            'iou': iou(torch.tensor(pred), torch.tensor(true))}
