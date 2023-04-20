import copy
import logging
import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
import sys
sys.path.append('.')

# from lib.datasets import heatmap, paf 
from lib.datasets.heatmap import putGaussianMaps
from lib.datasets.paf import putVecMaps

from lib.datasets import transformstool, utils

import glob
import json


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('node0'), keypoints.index('node1')],  
        [keypoints.index('node1'), keypoints.index('node2')],
        [keypoints.index('node1'), keypoints.index('node3')]
    ]
    return kp_lines
    
def get_keypoints():
    """Get the Surgicaltool keypoints and their left/right flip coorespondence map."""

    keypoints = [
        'node0',
        'node1',
        'node2',
        'node3'
    ]
    return keypoints

def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas

def collate_multiscale_images_anns_meta(batch):
    """Collate for multiscale.

    indices:
        images: [scale, batch , ...]
        anns: [batch, scale, ...]
        metas: [batch, scale, ...]
    """
    n_scales = len(batch[0][0])
    images = [torch.utils.data.dataloader.default_collate([b[0][i] for b in batch])
              for i in range(n_scales)]
    anns = [[b[1][i] for b in batch] for i in range(n_scales)]
    metas = [b[2] for b in batch]
    return images, anns, metas

def collate_images_targets_meta(batch):

    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets1 = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    targets2 = torch.utils.data.dataloader.default_collate([b[2] for b in batch])    

    return images, targets1, targets2


class Surgicaltool(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_transform=None, target_transforms=None, preprocess=None, input_y=368, input_x=368, stride=8):
        self.root_dir = root_dir 
        # self.image_files = sorted(glob.glob(os.path.join(root_dir,'**', '1', '**','raw.png'), recursive=True))
        self.ids = []
        
        # for index in range(len(self.image_files)):
        for index in range(len(self.root_dir)):
            self.ids.append(index)
        # print("ids = {0}".format(self.ids))

        # 处理器
        self.preprocess = preprocess  or transformstool.Normalize()
        self.image_transform = image_transform or transformstool.image_transform
        self.target_transforms = target_transforms

        # 热图
        self.HEATMAP_COUNT = len(get_keypoints())
        self.TOOL_IDS = kp_connections(get_keypoints())
        self.input_y = input_y
        self.input_x = input_x        
        self.stride = stride
        self.log = logging.getLogger(self.__class__.__name__)


    def __getitem__(self, idx):
        # print("images_files is.{0}".format(self.image_files))
        image_id = self.ids[idx]
        # img_path = self.image_files[idx]
        img_path = self.root_dir[idx]
        annotation_path = img_path.replace('raw.png', 'raw.json')

        # img_example = self.image_files[0]
        # self.log.debug(img_example)
        img_example = self.root_dir[0]
        self.log.debug(img_example)

        # Load image
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
        # Load annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        # return img, annotation 
        
        # Convert annotation to tensor
        # annotation = torch.tensor(annotation, dtype=torch.float32)
        
        # return img, annotation
        meta_init = {
            'dataset_index': idx,
            # file_name：coco的图像文件名
            'image_id': image_id,
            'file_name': 'raw',
        }

        img, annotation, meta = self.preprocess(img, annotation, None)
             
        if isinstance(img, list):
            return self.multi_image_processing(img, annotation, meta, meta_init)

        return self.single_image_processing(img, annotation, meta, meta_init)

        
    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        # 用于更新图像的元信息
        meta.update(meta_init) 

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        heatmaps, pafs = self.get_ground_truth(anns)
        
        heatmaps = torch.from_numpy(
            heatmaps.transpose((2, 0, 1)).astype(np.float32))
            
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))       
        return image, heatmaps, pafs

    # 去除非法关键点
    def remove_illegal_joint(self, keypoints):

        MAGIC_CONSTANT = (-1, -1, 0)
        print("remove_illegal_joint_keypoints = {0}".format(keypoints))
        mask = np.logical_or.reduce((keypoints[:, :, 0] >= self.input_x,
                                     keypoints[:, :, 0] < 0,
                                     keypoints[:, :, 1] >= self.input_y,
                                     keypoints[:, :, 1] < 0))
        keypoints[mask] = MAGIC_CONSTANT

        return keypoints

    def keypointtool(self, keypoint):
        # 'node0',            # 0
        # 'node1',            # 1
        # 'node2',            # 2
        # 'node3',            # 3
        our_order = [1,0,2,3]
        keypoint = keypoint[our_order, :]

        return keypoint


    def get_ground_truth(self, anns):
    
        grid_y = int(self.input_y / self.stride)
        grid_x = int(self.input_x / self.stride)
        channels_heat = (self.HEATMAP_COUNT + 1)
        channels_paf = 2 * len(self.TOOL_IDS)
        heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
        pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

        keypoints = []
        for ann in anns:
            # single_keypoints = np.array(ann['key_points']).reshape(17,3)
            single_keypoints = np.array(ann['keypoints']).reshape(4,3)
            single_keypoints = self.keypointtool(single_keypoints)
            keypoints.append(single_keypoints)
        keypoints = np.array(keypoints)
        keypoints = self.remove_illegal_joint(keypoints)

        # confidance maps for body parts
        for i in range(self.HEATMAP_COUNT):
            joints = [jo[i] for jo in keypoints]
            for joint in joints:
                if joint[2] > 0.5:
                    center = joint[:2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map,
                        7.0, grid_y, grid_x, self.stride)
        # pafs
        for i, (k1, k2) in enumerate(self.TOOL_IDS):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            for joint in keypoints:
                if joint[k1, 2] > 0.5 and joint[k2, 2] > 0.5:
                    centerA = joint[k1, :2]
                    centerB = joint[k2, :2]
                    vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                    pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                        centerA=centerA,
                        centerB=centerB,
                        accumulate_vec_map=vec_map,
                        count=count, grid_y=grid_y, grid_x=grid_x, stride=self.stride
                    )

        # background
        heatmaps[:, :, -1] = np.maximum(
            1 - np.max(heatmaps[:, :, :self.HEATMAP_COUNT], axis=2),
            0.
        )
        return heatmaps, pafs


    def __len__(self):
        return len(self.ids)
        # return len(self.image_files)
        # return len(self.root_dir)
        


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None, image_transform=None):
        self.image_paths = image_paths
        self.image_transform = image_transform or transforms.image_transform
        self.preprocess = preprocess

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.preprocess is not None:
            image = self.preprocess(image, [], None)[0]

        original_image = torchvision.transforms.functional.to_tensor(image)
        image = self.image_transform(image)

        return image_path, original_image, image

    def __len__(self):
        return len(self.image_paths)


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, image_transform=None):
        self.images = images
        self.image_transform = image_transform or transforms.image_transform

    def __getitem__(self, index):
        pil_image = self.images[index].copy().convert('RGB')
        original_image = torchvision.transforms.functional.to_tensor(pil_image)
        image = self.image_transform(pil_image)

        return index, original_image, image

    def __len__(self):
        return len(self.images)




# instance = Surgicaltool('/nfs/home/zhan/code/PoseEstimate/pytorch_Realtime_SurgicalTool_Pose_Estimation/data/robustmislite')
# print("images number", len(instance))
# img, annotation = instance[0]
# # cv2.imwrite('img_testxxxx.png',img)
# img.save("img_testxxxx.jpg")
# print("annotation = {0}".format(annotation))