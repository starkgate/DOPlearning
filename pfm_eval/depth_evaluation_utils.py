import configparser
import json

import numpy as np
import os
from path import Path
from cv2 import imread
from tqdm import tqdm


class test_framework_stillbox(object):
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=80, step=1):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.gt_files, self.img_files, self.displacements = read_scene_data(root, test_files, seq_length, step)

    def __getitem__(self, i):
        tgt = imread(self.img_files[i][0]).astype(np.float32)
        return {'tgt': tgt,
                'ref': [imread(img).astype(np.float32) for img in self.img_files[i][1]],
                'path': self.img_files[i][0],
                }

    def __len__(self):
        return len(self.img_files)


def get_displacements(scene, index, ref_indices):
    speed = np.around(np.linalg.norm(scene['speed']), decimals=3)
    assert(all(i < scene['length'] and i >= 0 for i in ref_indices)), str(ref_indices)
    return [speed*scene['time_step']*abs(index - i) for i in ref_indices]


def read_scene_data(data_root, test_list, seq_length=3, step=1):
    config = configparser.ConfigParser()
    config.read(os.path.join(data_root, 'seqinfo.ini'))

    im_files = []
    # how many frames around the current (tgt) frame should be taken into account by the network
    demi_length = (seq_length - 1) // 2
    # by default 1 frame before and 1 after: [-1, 1]
    shift_range = [step*i for i in list(range(-demi_length, 0)) + list(range(1, demi_length + 1))]

    for index, sample in enumerate(tqdm(test_list)):
        if os.path.isfile(sample):
            # clamp indices between 1 and the maximum number of frames in the scene
            capped_indices_range = list(map(lambda x: min(max(0, index + x), int(config['Sequence']['seqLength']) - 1), shift_range))
            ref_imgs_path = [test_list[ref_index] for ref_index in capped_indices_range]
            im_files.append([sample, ref_imgs_path])
        else:
            print('{} missing'.format(sample))

    return None, im_files, None


def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop gt to exclude border values
    # if used on gt_size 100x100 produces a crop of [-95, -5, 5, 95]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.05 * gt_height, 0.95 * gt_height,
                     0.05 * gt_width,  0.95 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask
