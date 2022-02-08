from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_SelfTraning(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-7
        self.max_num_epochs = 1
        self.num_batches_per_epoch = 1

    def maybe_update_lr(self, epoch=None):
        self.optimizer.param_groups[0]['lr'] = self.initial_lr
        self.print_to_log_file("lr:", self.optimizer.param_groups[0]['lr'])
