#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd


def export_for_submission(source_dir, target_dir):
    """
    promise wants mhd :-/
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    target_files = [join(target_dir, i[:-7] + ".mhd") for i in files]
    maybe_mkdir_p(target_dir)
    for f, t in zip(files, target_files):
        img = sitk.ReadImage(join(source_dir, f))
        sitk.WriteImage(img, t)


if __name__ == "__main__":
    folder = "/home/lvwei/enclave/public_datasets/LNDb"
    out_folder = "/home/lvwei/project/MedicalSegmentation/data/nnUNet_raw_data_base/nnUNet_raw_data/Task666_LNDb"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))
    # train
    current_dir = folder

    raw_data = [i for i in subfiles(current_dir, suffix="mhd")]
    # train & test label
    segmentations = subfiles(current_dir + "/masks", suffix="mhd")
    segmentations_group = {}
    for i in segmentations:
        if i.split("/")[-1][:9] not in segmentations_group.keys():
            segmentations_group[i.split("/")[-1][:9]] = [i]
        else:
            segmentations_group[i.split("/")[-1][:9]].append(i)

    remove_list = []
    pbar = tqdm(segmentations_group.items())
    for seg_name, seg_list in pbar:
        pbar.set_description("Processing Mask: %s" % seg_name)
        out_fname = join(out_folder, "labelsTr", seg_name + ".nii.gz")
        res_list = [sitk.ReadImage(i) for i in seg_list]
        for i in range(1, len(res_list)):
            res_list[0] += res_list[i]
        res_list[0] = (res_list[0] > 0)
        mask_sum = sitk.GetArrayViewFromImage(res_list[0]).sum()
        if mask_sum > 9:
            sitk.WriteImage(res_list[0], out_fname)
        else:
            raw_data.remove(f"{raw_data[0][:-13]}{seg_name}.mhd")
            remove_list.append({"removed_seg_name": seg_name, "mask_sum": mask_sum})
    remove_df = pd.DataFrame(remove_list)
    print(remove_df)
    remove_df.to_csv(join(out_folder, "removed_seg.csv"))
    assert len(raw_data) == len(segmentations_group) - len(remove_list), "Image and Mask number mismatch!"

    raw_data_train, raw_data_test = train_test_split(raw_data, test_size=0.1, random_state=20221031)

    # train
    pbar = tqdm(raw_data_train)
    for i in pbar:
        pbar.set_description("Processing Train Image: %s" % i)
        out_fname = join(out_folder, "imagesTr", i.split("/")[-1][:-4] + "_0000.nii.gz")
        sitk.WriteImage(sitk.ReadImage(i), out_fname)
    # test
    pbar = tqdm(raw_data_test)
    for i in pbar:
        pbar.set_description("Processing Test Image: %s" % i)
        out_fname = join(out_folder, "imagesTs", i.split("/")[-1][:-4] + "_0000.nii.gz")
        sitk.WriteImage(sitk.ReadImage(i), out_fname)



    json_dict = OrderedDict()
    json_dict['name'] = "LNDb"
    json_dict['description'] = "nodule with merger all rad seg (union)"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "nodule"
    }
    json_dict['numTraining'] = len(raw_data_train)
    json_dict['numTest'] = len(raw_data_test)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-4], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-4]} for i in raw_data_train]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-4] for i in raw_data_test]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

