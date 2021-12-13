from utils import generate_dataset_json
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def put_file_in_place():
    filedf = pd.read_csv("../data/LIDC-IDRI/LIDCIDRI_Available_GESE.csv")
    filedf = filedf[filedf["Manufacturer"] == "SIEMENS"]
    traindf, testdf = train_test_split(filedf, test_size=0.2, random_state=0)
    cnt = 1
    for idx, row in traindf.iterrows():
        img_filename = "LI_" + "0" * (3 - len(str(cnt))) + str(cnt) + "_0000.nii.gz"
        mask_filename = "LI_" + "0" * (3 - len(str(cnt))) + str(cnt) + ".nii.gz"
        if os.path.isfile(os.path.join("../data/LIDC-IDRI/LIDC_IDRI_NII/imgs", row["Subject ID"] + ".nii.gz")):
            os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/imgs", row["Subject ID"] + ".nii.gz"),
                       os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/imagesTr", img_filename))
            os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/labels", row["Subject ID"] + "_mask.nii.gz"),
                       os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/labelsTr", mask_filename))
        else:
            os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/imgs", row["Subject ID"] + row["Study UID"] + ".nii.gz"),
                       os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/imagesTr", img_filename))
            os.symlink(os.path.join("/home/lvwei/project/MedicalSegmentation/data/LIDC-IDRI/LIDC_IDRI_NII/labels", row["Subject ID"] + row["Study UID"] + "_mask.nii.gz"),
                       os.path.join("../data/LIDC-IDRI/Task555_LungNoduleSE/labelsTr", mask_filename))
        cnt += 1

if __name__ == '__main__':
    put_file_in_place()
