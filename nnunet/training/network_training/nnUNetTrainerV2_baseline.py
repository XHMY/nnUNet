import torch

from nnunet.network_architecture.generic_modular_UNet import get_default_network_config, PlainConvUNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import \
    nnUNetTrainerV2_ResencUNet
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainerV2_Baseline(nnUNetTrainerV2_ResencUNet):
    def initialize_network(self):
        if self.threeD:
            cfg = get_default_network_config(3, None, norm_type="in")

        else:
            cfg = get_default_network_config(1, None, norm_type="in")

        stage_plans = self.plans['plans_per_stage'][self.stage]
        conv_kernel_sizes = stage_plans['conv_kernel_sizes']
        blocks_per_stage_encoder = stage_plans['num_blocks_encoder']
        blocks_per_stage_decoder = stage_plans['num_blocks_decoder']
        pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        self.network = PlainConvUNet(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                                   pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                                   blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
