from nnunet.training.network_training import nnUNetTrainerV2

class nnUNetTrainerV2_TransferLess(nnUNetTrainerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 50
        self.initial_lr = 1e-4
        self.num_batches_per_epoch = 50