import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
import netCDF4


class Tem_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path,training=True):
        super(Tem_Dataset, self).__init__()
        self.training = training
        self.data = self.load_list(image_path)#获得路径
        print(image_path)
        print(len(self.data))
        # self.data = list(glob.glob(image_path+'/*.nc')).sort()  # 获得路径
        # print(self.data)
        #self.mask_data = self.load_list(mask_path)

        self.mask_data = list(glob.glob(mask_path+'/*.h5'))#获取mask路径
        self.mask_data.sort()
        print(len(self.mask_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    #打开图片和mask，修改为打开.nc文件
    def load_item(self, index):
        #温度数据
        temp = netCDF4.Dataset('{:s}'.format(self.data[index]))  # 打开.nc文件
        temp_data = temp['skt'][...]
        temp_data = np.array(temp_data[100:, :])#[0,100:, :]
        #avg = np.mean(temp_data)
        temp_data = torch.from_numpy(temp_data).float()  # 将数据转换为tensor
        temp_data = temp_data.unsqueeze(0)
        temp_data = torch.repeat_interleave(temp_data, repeats=3, dim=0)  # 格式为C，H，W
        temp.close()

        #mask
        mask = netCDF4.Dataset(self.mask_data[index])
        mask_data = mask['skt'][...]
        mask_data = np.array(mask_data[100:, :])
        mask_data = torch.from_numpy(mask_data).float()  # 将数据转换为tensor
        mask_data = mask_data.unsqueeze(0)
        if self.training:
            mask_data = torch.repeat_interleave(mask_data, repeats=3, dim=0)
        mask.close()
        return temp_data, mask_data

    #此函数主要负责获得数据的路径
    def load_list_mask(self, path):
        # 改为.nc文件
        path = list(glob.glob(path+'/*.h5'))
        path.sort()
        return path
    def load_list(self, path):
        # 改为.nc文件
        path = list(glob.glob(path+'/*.nc'))
        path.sort()
        return path
