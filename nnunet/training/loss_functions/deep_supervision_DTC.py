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


class MultipleOutputLoss2DTC(nn.Module):
    def __init__(self, seg_loss, weight_factors=None, consistency=1.0, consistency_rampup=150.0):
        """
        for dual task output
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param seg_loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2DTC, self).__init__()
        self.weight_factors = weight_factors
        self.seg_loss = seg_loss
        self.lsf_loss = nn.MSELoss()
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.epoch = 0

    def consistency_loss(self, seg_output, lsf_output):
        return self.lsf_loss(seg_output, torch.sigmoid(-1500 * lsf_output))  # Question: why don't we use dice loss
        # return self.seg_loss(seg_output, torch.sigmoid(-1500 * lsf_output))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_current_consistency_weight(self):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        def sigmoid_rampup(current, rampup_length):
            if rampup_length == 0:
                return 1.0
            else:
                current = np.clip(current, 0.0, rampup_length)
                phase = 1.0 - current / rampup_length
                return float(np.exp(-5.0 * phase * phase))

        return self.consistency * sigmoid_rampup(self.epoch, self.consistency_rampup)

    def forward(self, output, target):
        assert isinstance(output, (tuple, list)), "x must be either tuple or list"
        assert isinstance(target, (tuple, list)), "y must be either tuple or list"
        assert len(output[0]) == len(output[1]), "deep supervision output must equal length"
        if self.weight_factors is None:
            weights = [1] * len(output[0])
        else:
            weights = self.weight_factors

        # assert target[0][:,1:2].min() == 0 and target[0][:,1:2].max() <= 1, "seg label min/max error"
        l_seg = weights[0] * self.seg_loss(output[1][0], target[0][:,1:2])
        l_lsf = weights[0] * self.lsf_loss(output[0][0], target[0][:,0:1])
        l_consis = weights[0] * self.consistency_loss(output[1][0][:, 1], output[0][0][:, 0])
        for i in range(1, len(output[0])):
            if weights[i] != 0:
                l_seg += weights[i] * self.seg_loss(output[1][i], target[i][:,1:2])
                l_lsf += weights[i] * self.lsf_loss(output[0][i], target[i][:,0:1])
                # l_consis += weights[i] * self.consistency_loss(output[1][i][:, 1], output[0][i][:, 0])
        return l_seg, l_lsf, l_consis, self.get_current_consistency_weight()



if __name__ == '__main__':
    pass
