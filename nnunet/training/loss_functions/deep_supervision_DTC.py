#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
import torch
from torch import nn


class DTCLoss_DTC(nn.Module):
    def __init__(self, seg_loss, weight_factors=None, consistency=1.0, consistency_rampup=40.0, is_unsupervised=False):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param seg_loss:
        :param weight_factors:
        """
        super(DTCLoss_DTC, self).__init__()
        self.weight_factors = weight_factors
        self.seg_loss = seg_loss
        self.mse_loss = nn.MSELoss()
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.cur_epochs = 0
        self.is_unsupervised = is_unsupervised

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242

        def sigmoid_rampup(current, rampup_length):
            if rampup_length == 0:
                return 1.0
            else:
                current = np.clip(current, 0.0, rampup_length)
                phase = 1.0 - current / rampup_length
                return float(np.exp(-5.0 * phase * phase))

        return self.consistency * sigmoid_rampup(epoch, self.consistency_rampup)

    def seg_deep_super_loss(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list. But got type " \
                                             + str(type(x)) + " shape " + str(x.shape)
        assert isinstance(y, (tuple, list)), "y must be either tuple or list" \
                                             + str(type(y)) + " shape " + str(y.shape)

        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.seg_loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.seg_loss(x[i], y[i])
        return l

    def lsf_loss(self, x, y):
        return self.mse_loss(x, y)

    def dtc_loss(self, x, y):
        return self.mse_loss(torch.sigmoid(-1500 * x), y)

    def forward(self, x, y):
        """
        param x: x[0] is regression output, x[1] is segmentation output in different layer
        """
        # print("Length of x,y:", len(x), len(y))
        # for xi, xc in enumerate(x[1]):
        #     print("Shape of x[1][" + str(xi) + "]:", xc.shape)
        # for yi, yc in enumerate(y[1]):
        #     print("Shape of y[1][" + str(yi) + "]:", yc.shape)
        #     print("Sum:", torch.sum(yc))

        unsupervised_loss = self.dtc_loss(x[1][0], x[0])
        if self.is_unsupervised:
            print("This is a unsupervised training.")
            return unsupervised_loss
        supervise_loss = self.seg_deep_super_loss(x[1], y[1]) + 0.3 * self.lsf_loss(x[0], y[0])
        consistency_weight = self.get_current_consistency_weight(self.cur_epochs)
        return supervise_loss + unsupervised_loss * consistency_weight


if __name__ == '__main__':
    pass
