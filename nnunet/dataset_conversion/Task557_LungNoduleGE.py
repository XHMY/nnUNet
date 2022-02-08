import os

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split

from utils import generate_dataset_json

raw_data_path = os.path.join(
    os.environ["nnUNet_raw_data_base"], "nnUNet_raw_data")
# dataset_path = "/data/home/scy0313/run/data/LIDC-IDRI/LIDC_IDRI_NII"
dataset_path = "/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII"
dataset_name = 'Task557_LungNoduleGE'


def make_link(cnt, row, tt):
    img_filename = "LI_" + "0" * \
                   (3 - len(str(cnt))) + str(cnt) + "_0000.nii.gz"
    mask_filename = "LI_" + "0" * (3 - len(str(cnt))) + str(cnt) + ".nii.gz"
    if os.path.isfile(os.path.join(dataset_path, "imgs", row["Subject ID"] + ".nii.gz")):
        os.symlink(os.path.join(dataset_path, "imgs", row["Subject ID"] + ".nii.gz"),
                   os.path.join(raw_data_path, dataset_name, tt, img_filename))
        os.symlink(os.path.join(dataset_path, "labels", row["Subject ID"] + "_mask.nii.gz"),
                   os.path.join(raw_data_path, dataset_name, "labelsTr", mask_filename))
    else:
        os.symlink(os.path.join(dataset_path, "imgs", row["Subject ID"] + row["Study UID"] + ".nii.gz"),
                   os.path.join(raw_data_path, dataset_name, tt, img_filename))
        os.symlink(os.path.join(dataset_path, "labels", row["Subject ID"] + row["Study UID"] + "_mask.nii.gz"),
                   os.path.join(raw_data_path, dataset_name, "labelsTr", mask_filename))


def put_file_in_place():
    filedf = pd.read_csv(os.path.join(
        dataset_path[:-13], "LIDCIDRI_Available_GESE.csv"))
    filedf = filedf[filedf["Manufacturer"] == "GE MEDICAL SYSTEMS"]

    try:
        os.makedirs(os.path.join(raw_data_path,
                                 dataset_name, "imagesTr"))
        os.makedirs(os.path.join(raw_data_path,
                                 dataset_name, "imagesTs"))
        os.makedirs(os.path.join(raw_data_path,
                                 dataset_name, "labelsTr"))
    except FileExistsError:
        pass

    traindf, testdf = train_test_split(filedf, test_size=0.1, random_state=0)
    cnt = 1
    for idx, row in traindf.iterrows():
        make_link(cnt, row, "imagesTr")
        cnt += 1
    for idx, row in testdf.iterrows():
        make_link(cnt, row, "imagesTs")
        cnt += 1


def gen_dataset_json():
    DATASET_PATH = os.path.join(raw_data_path, dataset_name)
    generate_dataset_json(os.path.join(DATASET_PATH, "dataset.json"),
                          os.path.join(DATASET_PATH, "imagesTr"),
                          os.path.join(DATASET_PATH, "imagesTs"),
                          ('CT',),
                          {0: 'background', 1: 'nodule'},
                          dataset_name,
                          dataset_description="LIDC-IDRI GE MEDICAL SYSTEMS"
                          )


def gen_splits_final_pkl():
    out_preprocessed = os.path.join(
        os.environ["nnUNet_raw_data_base"], dataset_name)
    splits = [{'train': ["LI_" + "0" * (3 - len(str(i))) + str(i) for i in range(
        1, 150)], 'val': ["LI_" + "0" * (3 - len(str(i))) + str(i) for i in range(150, 188)]}]
    # print(splits)
    save_pickle(splits, join(out_preprocessed, "splits_final.pkl"))


if __name__ == '__main__':
    put_file_in_place()
    gen_dataset_json()
    pass
