import torchIoULoss
import torch.nn as nn
import torch.nn.functional as F
import torch
class IoUScore(nn.Module):
    def __init__(self, n_classes=2):
        super(IoUScore, self).__init__()
        self.n_classes = n_classes

   

    def forward(self, input1, input2):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input1)

        # Numerator Product
        inter = input1 * input2
      
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = input1 + input2 - (input1 * input2)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        rw = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return rw.mean()
