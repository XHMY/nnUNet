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
from typing import Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast

from nnunet.network_architecture.generic_modular_DTC_UNet import DTCUNet, get_default_network_config
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.dataloading.dataset_loading import DataLoader3D
from nnunet.training.loss_functions.deep_supervision_DTC import MultipleOutputLoss2DTC
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class nnUNetTrainerV2DTC(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.consis_weight = 0.3
        self.lsf_weight = 0.3
        # self.max_num_epochs = 1 # For Test Only

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training=training, force_load_plans=force_load_plans)
        self.loss = MultipleOutputLoss2DTC(seg_loss=self.loss.loss, weight_factors=self.loss.weight_factors)

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

        self.network = DTCUNet(self.num_input_channels, self.base_num_features, blocks_per_stage_encoder, 2,
                               pool_op_kernel_sizes, conv_kernel_sizes, cfg, self.num_classes,
                               blocks_per_stage_decoder, True, False, 320, InitWeights_He(1e-2))
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        net_num_pool_op_kernel_sizes is different in resunet
        """
        super().setup_DA_params()
        self.data_aug_params['selected_seg_channels'] = None
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes[1:]), axis=0))[:-1]

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        return nnUNetTrainer.run_online_evaluation(self, output[1][0], target[0][:,1::1])

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[
        np.ndarray, np.ndarray]:
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        self.network.decoder.compute_level_set_regression = False
        ret = nnUNetTrainer.predict_preprocessed_data_return_seg_and_softmax(self, data, do_mirroring=do_mirroring,
                                                                             mirror_axes=mirror_axes,
                                                                             use_sliding_window=use_sliding_window,
                                                                             step_size=step_size,
                                                                             use_gaussian=use_gaussian,
                                                                             pad_border_mode=pad_border_mode,
                                                                             pad_kwargs=pad_kwargs,
                                                                             all_in_gpu=all_in_gpu,
                                                                             verbose=verbose,
                                                                             mixed_precision=mixed_precision)
        self.network.decoder.deep_supervision = ds
        self.network.decoder.compute_level_set_regression = True
        return ret

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        ret = nnUNetTrainer.run_training(self, enable_dtc=True)
        self.network.decoder.deep_supervision = ds
        self.loss.set_epoch(self.epoch)
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        assert self.fp16, "only implemented for fp16"

        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        # print("data.shape:", data.shape)
        # print("target length:", len(target))
        # print("target[0].shape", target[0].shape)
        # for yi, yc in enumerate(target[1]):
        #     print("Shape of y[1][" + str(yi) + "]:", yc.shape)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        with autocast():
            output = self.network(data)
            del data
            l_seg, l_lsf, l_consis, rampup_consistency_weight = self.loss(output, target)
            l = (1 - self.consis_weight) * ((1 - self.lsf_weight) * l_seg + self.lsf_weight * l_lsf) + \
                self.consis_weight * rampup_consistency_weight * l_consis

        if do_backprop:
            self.amp_grad_scaler.scale(l).backward()
            self.amp_grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.amp_grad_scaler.step(self.optimizer)
            self.amp_grad_scaler.update()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        self.loss_detail_log_sum["seg_loss"].append(l_seg.detach().cpu().numpy())
        self.loss_detail_log_sum["lsf_loss"].append(l_lsf.detach().cpu().numpy())
        self.loss_detail_log_sum["consis_loss"].append(l_consis.detach().cpu().numpy())

        del target

        return l.detach().cpu().numpy()

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = False
        ret = nnUNetTrainer.validate(self, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window,
                                     step_size=step_size,
                                     save_softmax=save_softmax, use_gaussian=use_gaussian,
                                     overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                                     all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                                     run_postprocessing_on_folds=run_postprocessing_on_folds, has_lsf=True)

        self.network.decoder.deep_supervision = ds
        return ret

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        assert self.threeD, "This function is only implemented for 3D data"

        dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                             False, has_level_set=True,
                             oversample_foreground_percent=self.oversample_foreground_percent,
                             pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                              has_level_set=True, oversample_foreground_percent=self.oversample_foreground_percent,
                              pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val
