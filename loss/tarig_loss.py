import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

def dice_loss(input, target, mask=None):
    if mask is not None:
        idxs = torch.nonzero(mask.flatten()).flatten()
        input = input[:, idxs]
        target = target[:, idxs]

    a = torch.sum(input * target)
    b = torch.sum(input * input) +  1e-5 
    c = torch.sum(target * target) + 1e-5
    d = (2 * a) / (b + c)
    return 1-d


def aux_dice_loss(input, target, skin_mask, mask=None):
    if mask is not None:
        idxs = torch.nonzero(mask.flatten()).flatten()
        input = input[:, idxs]
        target = target[:, idxs]
        skin_mask = skin_mask[:, idxs]

    input = input[skin_mask]
    target = target[skin_mask]
    a = torch.sum(input * target)
    b = torch.sum(input * input) +  1e-5 
    c = torch.sum(target * target) + 1e-5 
    d = (2 * a) / (b + c)
    return 1-d


def log_softmax_loss(input, target):
    score = F.softmax(input, dim=1)
    err = -target * torch.log(score + 1e-5)
    loss = err.sum(dim=1).mean()
    return loss


def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    return sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma, reduction="mean")


def mse_loss(conflow_pred, conflow_gt, attn=None, reduction="mean"):
    if attn is not None:
        mse = torch.sum((conflow_pred[attn, :] - conflow_gt[attn, :]) ** 2, dim=1)
    else:
        mse = torch.sum((conflow_pred - conflow_gt) ** 2, dim=1)
    loss = mse.mean()
    return loss
