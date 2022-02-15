import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from utils import generate_dataset_json

raw_data_path = os.path.join(os.environ["nnUNet_raw_data_base"], "nnUNet_raw_data")
dataset_name = 'Task558_LungNoduleGESemi'
dataset_path = "/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII"


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
    file_df = pd.read_csv(os.path.join(
        dataset_path[:-13], "LIDCIDRI_Available_GESE.csv"))
    file_df = file_df[file_df["Manufacturer"] == "GE MEDICAL SYSTEMS"]
    nodule_cnt_df = pd.read_excel(os.path.join(dataset_path[:-13], "ORI_meta/lidc-idri nodule counts (6-23-2015).xlsx"))[
        ["Number of Nodules >=3mm**", "TCIA Patent ID"]].dropna()
    file_df = pd.merge(file_df, nodule_cnt_df, left_on="Subject ID", right_on="TCIA Patent ID", how="left")
    file_df = file_df[file_df["Number of Nodules >=3mm**"] > 1].sample(n=50, random_state=123).reset_index(drop=True)

    try:
        os.makedirs(os.path.join(raw_data_path, dataset_name))
        os.makedirs(os.path.join(raw_data_path,
                                 dataset_name, "imagesTr"))
        os.makedirs(os.path.join(raw_data_path,
                                 dataset_name, "imagesTs"))
        os.makedirs(os.path.join(raw_data_path,
                                 dataset_name, "labelsTr"))
    except FileExistsError:
        print("Folder already exists, use them")
        pass

    cnt = 1
    for idx, row in file_df.iterrows():
        make_link(cnt, row, "imagesTr")
        cnt += 1


def gen_dataset_json():
    DATASET_PATH = os.path.join(raw_data_path, dataset_name)
    generate_dataset_json(os.path.join(DATASET_PATH, "dataset.json"),
                          os.path.join(DATASET_PATH, "imagesTr"),
                          os.path.join(DATASET_PATH, "imagesTs"),
                          ('CT',),
                          {0: 'background', 1: 'nodule'},
                          dataset_name,
                          dataset_description="LIDC-IDRI GE MEDICAL SYSTEMS Test for Semi-supervised learning"
                          )




if __name__ == '__main__':
    put_file_in_place()
    gen_dataset_json()
