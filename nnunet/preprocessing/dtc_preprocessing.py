import os
import pickle
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np

from nnunet.configuration import default_num_threads
from nnunet.preprocessing.preprocessing import GenericPreprocessor


def compute_sdf(img_gt):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    assert img_gt.min() == 0, "img_gt.min() = " + str(img_gt.min())
    img_gt = img_gt.astype(np.uint8)
    posmask = img_gt.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        boundary = skimage_seg.find_boundaries(posmask, mode='outer').astype(np.uint8)
        sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                np.max(posdis) - np.min(posdis))
        sdf[boundary == 1] = 0
        assert np.min(sdf) == -1.0, (np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
        assert np.max(sdf) ==  1.0, (np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return sdf


class DTCPreprocessor(GenericPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run_internal(self, target_spacing, case_identifier, output_folder_stage, cropped_output_dir, force_separate_z,
                      all_classes):
        data, seg, properties = self.load_cropped(cropped_output_dir, case_identifier)

        data = data.transpose((0, *[i + 1 for i in self.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in self.transpose_forward]))

        data, seg, properties = self.resample_and_normalize(data, target_spacing,
                                                            properties, seg, force_separate_z)
        try:
            lsf_value = compute_sdf(seg)
        except AssertionError:
            np.savez(os.path.join('/', *output_folder_stage.split('/')[:-1], "debug_output", case_identifier + "_seg.npz"), data=seg)
            np.savez(os.path.join('/', *output_folder_stage.split('/')[:-1], "debug_output", case_identifier + "_img.npz"), data=data)
            raise AssertionError(case_identifier + " Level Set Function Compute Error.")
        all_data = np.vstack((data, lsf_value, seg)).astype(np.float32)

        # we need to find out where the classes are and sample some random locations
        # let's do 10.000 samples per class
        # seed this for reproducibility!
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(1234)
        class_locs = {}
        for c in all_classes:
            all_locs = np.argwhere(all_data[-1] == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            print(c, target_num_samples)
        properties['class_locations'] = class_locs

        print("saving: ", os.path.join(output_folder_stage, "%s.npz" % case_identifier))
        np.savez_compressed(os.path.join(output_folder_stage, "%s.npz" % case_identifier),
                            data=all_data.astype(np.float32))
        with open(os.path.join(output_folder_stage, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        target_spacings = [target_spacings[0]]  # only stage 0 is needed
        num_threads = [num_threads[0]]  # only stage 0 is needed
        super(DTCPreprocessor, self).run(target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
                                         num_threads, force_separate_z)

if __name__ == '__main__':
    pass