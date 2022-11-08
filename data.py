import os
import torch
from PIL import Image
import glob
import random
import queue
import threading
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


class HazeDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, haze_root, transforms, upscale_factor):
        self.upscale_factor = upscale_factor
        self.haze_root = haze_root
        self.ori_root = ori_root
        self.image_name_list = glob.glob(os.path.join(self.haze_root, '*.jpg'))
        self.matching_dict = {}
        self.file_list = []
        self.get_image_pair_list()
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, haze_img_lr, ori_img
        """
        ori_image_name, haze_image_name = self.file_list[item]
        ori_image = self.transforms(Image.open(ori_image_name))
        haze_image = self.transforms(Image.open(haze_image_name))

        w = haze_image.shape[1]// self.upscale_factor
        h = haze_image.shape[2]// self.upscale_factor
        lr_scale = Resize([w, h] , interpolation=Image.BICUBIC)
        haze_img_lr = lr_scale(haze_image)
        
        return ori_image, haze_image, haze_img_lr

    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            image = image.split("/")[-1]
            if os.name == 'nt':
                image = image.split("\\")[-1]
                
            key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
            if key in self.matching_dict.keys():
                self.matching_dict[key].append(image)
            else:
                self.matching_dict[key] = []
                self.matching_dict[key].append(image)

        for key in list(self.matching_dict.keys()):
            for hazy_image in self.matching_dict[key]:
                self.file_list.append([os.path.join(self.ori_root, key), os.path.join(self.haze_root, hazy_image)])

        random.shuffle(self.file_list)

