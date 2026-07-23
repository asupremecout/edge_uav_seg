import torch
import torch.nn as nn

def dice_loss(logits, targets, smooth=1e-6):
    """
    Compute the Dice loss between predicted and target tensors.

    Args:
        logits (torch.Tensor): Predicted logits of shape (N, C, H, W).
        targets (torch.Tensor): Ground truth tensor of shape (N, C, H, W).
        smooth (float): Smoothing factor to avoid division by zero.
            Default is 1e-6.
    Returns:
        torch.Tensor: The computed Dice loss.
    """
    num_classes = logits.shape[1]

    probs=torch.softmax(logits, dim=1)

    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    intersection = torch.sum(probs * targets_one_hot, dim=( 2, 3))
    union=probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score.mean()

    return dice_loss


    