import os

from batchgenerators.utilities.file_and_folder_operations import *

from utils import generate_dataset_json

raw_data_path = os.path.join(os.environ["nnUNet_raw_data_base"], "nnUNet_raw_data")
pred_path = '/home/lvwei/project/MedicalSegmentation/data/Pred/Task557_LungNoduleGE'
dataset_name = 'Task558_LungNoduleGESemi'


def gen_dataset_json():
    DATASET_PATH = os.path.join(raw_data_path, dataset_name)
    generate_dataset_json(os.path.join(DATASET_PATH, "dataset.json"),
                          os.path.join(DATASET_PATH, "imagesTr"),
                          os.path.join(DATASET_PATH, "imagesTs"),
                          ('CT',),
                          {0: 'background', 1: 'nodule'},
                          dataset_name,
                          dataset_description="LIDC-IDRI GE MEDICAL SYSTEMS Test for Self-Traning (3d_lowres 554)"
                          )


def gen_splits_final_pkl():
    out_preprocessed = os.path.join(
        os.environ["nnUNet_raw_data_base"], "nnUNet_raw_data", dataset_name)
    splits = [{'train': ["LI_" + "0" * (3 - len(str(i))) + str(i) for i in range(502, 558)], 'val': []}]
    print(splits)
    save_pickle(splits, join(out_preprocessed, "splits_final.pkl"))


if __name__ == '__main__':
    gen_dataset_json()
    gen_splits_final_pkl()
