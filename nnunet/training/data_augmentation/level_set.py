import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class LevelSetTransform(AbstractTransform):
    def __init__(self, input_key="target", output_key="target"):
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, **data_dict):
        """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                 -inf|x-y|; x in segmentation
                 +inf|x-y|; x out of segmentation
        normalize sdf to [-1,1]
        """
        img_gt = data_dict[self.input_key][0]
        img_gt = img_gt.astype(np.uint8)
        out_shape = img_gt.shape
        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]):  # batch size
            posmask = img_gt[b].astype(bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                        np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

        data_dict[self.output_key] = (normalized_sdf, data_dict[self.input_key])
        return data_dict


# if __name__ == '__main__':
#     from batchviewer import view_batch
#
#     np.random.seed(123)
#     image_gt = np.random.randint(0, 2, size=(3, 30, 300, 300))
#     res = compute_level_set(image_gt, image_gt.shape)
#     view_batch(image_gt, width=300, height=300)
#     view_batch(res, width=300, height=300)
#     print(res.shape)
