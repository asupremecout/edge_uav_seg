from losses.dice_loss import dice_loss
from losses.focal_loss import focal_loss

def conbined_loss(logits, targets, alpha=0.5, gamma=2.0, smooth=1e-6):
    """
    Compute the combined loss of Dice loss and Focal loss.

    Args:
        logits (torch.Tensor): Predicted logits of shape (N, C, H, W).
        targets (torch.Tensor): Ground truth tensor of shape (N, H, W).
        alpha (float): Weight for Dice loss in the combined loss. Default is 0.5.
        gamma (float): Focusing parameter for Focal loss. Default is 2.0.
        smooth (float): Smoothing factor for Dice loss to avoid division by zero. Default is 1e-6.

    Returns:
        torch.Tensor: The computed combined loss.
    """
    dice = dice_loss(logits, targets, smooth)
    focal = focal_loss(logits, targets, gamma)

    combined = alpha * dice + (1 - alpha) * focal
    return combined