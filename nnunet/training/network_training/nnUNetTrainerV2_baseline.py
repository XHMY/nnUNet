import torch

from nnunet.network_architecture.generic_modular_UNet import get_default_network_config, PlainConvUNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.data_augmentation.default_data_augmentation_DS import get_default_augmentation_DTC_DS
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNet_variants.architectural_variants.nnUNetTrainerV2_ResencUNet import \
    nnUNetTrainerV2_ResencUNet
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainerV2_Baseline_defaultDA(nnUNetTrainerV2_ResencUNet):

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        self.tr_gen, self.val_gen = get_default_augmentation_DTC_DS(
            self.dl_tr, self.dl_val,
            self.data_aug_params[
                'patch_size_for_spatialtransform'],
            self.data_aug_params,
            deep_supervision_scales=self.deep_supervision_scales,
            pin_memory=self.pin_memory
        )

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.95, nesterov=True)
        self.lr_scheduler = None

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


    def on_epoch_end(self):
        continue_training = nnUNetTrainer.on_epoch_end(self)
        return continue_training and self.epoch < self.max_num_epochs