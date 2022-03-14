import os
import shutil
from os.path import join

import nnunet
from nnunet.configuration import default_num_threads
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.training.model_restore import recursive_find_python_class


class ExperimentPlanner3D_DTC_v21(ExperimentPlanner3D_v21):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor_name = "DTCPreprocessor"