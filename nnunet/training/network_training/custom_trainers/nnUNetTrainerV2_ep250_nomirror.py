from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_ep250_nomirror(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
                         
        self.max_num_epochs = 250

    def setup_DA_params(self):
        nnUNetTrainerV2.setup_DA_params(self)
        self.data_aug_params["do_mirror"] = False
        # from pprint import pprint
        # pprint(self.data_aug_params)
