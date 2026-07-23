import torch
import torch.nn as nn

def focal_loss(logits,targets,gamma=2.0,alpha=None):

    ce_loss=nn.CrossEntropyLoss(reduction='none')(logits,targets)

    probs=torch.softmax(logits,dim=1)

    p_t=torch.gather(probs,dim=1,index=targets.unsqueeze(1)).squeeze(1)

    focal_weight=(1.0-p_t)**gamma

    return (focal_weight*ce_loss).mean()