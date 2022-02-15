from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerV2_DTC import nnUNetTrainerV2DTC


class nnUNetTrainerV2DTC_Unsupervised(nnUNetTrainerV2DTC):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-6
        self.max_num_epochs = 1

    def maybe_update_lr(self, epoch=None):
        self.optimizer.param_groups[0]['lr'] = 1e-6
        self.print_to_log_file("lr:", self.optimizer.param_groups[0]['lr'])

    def run_training(self):
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.decoder.deep_supervision
        self.network.decoder.deep_supervision = True
        self.loss.is_unsupervised = True
        ret = nnUNetTrainer.run_training(self)
        self.network.decoder.deep_supervision = ds
        return ret
