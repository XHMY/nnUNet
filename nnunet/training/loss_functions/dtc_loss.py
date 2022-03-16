from torch import nn


class DTCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        return nn.functional.binary_cross_entropy_with_logits(input, target, weight=self.weight)