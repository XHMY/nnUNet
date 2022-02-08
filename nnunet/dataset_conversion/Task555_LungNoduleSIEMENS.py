import os

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split

from utils import generate_dataset_json


def make_link(cnt, row, tt):
    img_filename = "LI_" + "0" * (3 - len(str(cnt))) + str(cnt) + "_0000.nii.gz"
    mask_filename = "LI_" + "0" * (3 - len(str(cnt))) + str(cnt) + ".nii.gz"
    if os.path.isfile(os.path.join("../data/LIDC-IDRI/LIDC_IDRI_NII/imgs", row["Subject ID"] + ".nii.gz")):
        os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/imgs",
                                row["Subject ID"] + ".nii.gz"),
                   os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/" + tt, img_filename))
        os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/labels",
                                row["Subject ID"] + "_mask.nii.gz"),
                   os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/labelsTr", mask_filename))
    else:
        os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/imgs",
                                row["Subject ID"] + row["Study UID"] + ".nii.gz"),
                   os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/" + tt, img_filename))
        os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/labels",
                                row["Subject ID"] + row["Study UID"] + "_mask.nii.gz"),
                   os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/labelsTr", mask_filename))


def put_file_in_place():
    filedf = pd.read_csv("../data/LIDC-IDRI/LIDCIDRI_Available_GESE.csv")
    filedf = filedf[filedf["Manufacturer"] == "SIEMENS"]
    traindf, testdf = train_test_split(filedf, test_size=0.1, random_state=0)
    cnt = 1
    for idx, row in traindf.iterrows():
        make_link(cnt, row, "imagesTr")
        cnt += 1
    for idx, row in testdf.iterrows():
        make_link(cnt, row, "imagesTs")
        cnt += 1


def gen_dataset_json():
    DATASET_PATH = "/home/lvwei/project/MedicalSegmentation/data/nnUNet_raw/nnUNet_raw_data/Task555_LungNoduleSE/"
    generate_dataset_json(DATASET_PATH + "dataset.json",
                          DATASET_PATH + "imagesTr",
                          DATASET_PATH + "imagesTs",
                          ('CT',),
                          {0: 'background', 1: 'nodule'},
                          "Task555_LungNoduleSE",
                          dataset_description="LIDC-IDRI SIEMENS"
                          )


def gen_splits_final_pkl():
    out_preprocessed = "/home/lvwei/project/MedicalSegmentation/data/nnUNet_preprocessed/Task555_LungNoduleSE"
    splits = [{'train': ["LI_" + "0" * (3 - len(str(i))) + str(i) for i in range(1, 150)],
               'val': ["LI_" + "0" * (3 - len(str(i))) + str(i) for i in range(150, 188)]}]
    # print(splits)
    save_pickle(splits, join(out_preprocessed, "splits_final.pkl"))


if __name__ == '__main__':
    # put_file_in_place()
    # gen_dataset_json()
    gen_splits_final_pkl()
    pass
