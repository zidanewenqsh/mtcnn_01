#!/usr/bin/env python
# D:\MySoft\Anaconda3 python
# coding:UTF-8
"""
@version: python3.7
@author:wenqsh
@contact:
@software: PyCharm
@file:
@title: mtcnndataset
@time: 2019/09/09 16:43
@result:
"""
# import os
# import torch
# import numpy as np
# import torch.utils.data as data
# from PIL import Image
# from torchvision import transforms
# import time
# import torch
from tool.utils import *


class Timekeeper:
    '''
    用来计算程序运行的时间
    '''
    _time = time.time()

    # _interval = 0
    def __new__(cls, *args, **kwargs):
        return super(Timekeeper, cls).__new__(cls)

    def __init__(self):
        pass

    @classmethod
    def gettime(cls):
        # cls._interval = time.time()-cls._time
        return time.time() - cls._time

    @classmethod
    def reinittime(cls):
        cls._time = time.time()


transform = transforms.Compose([
    # transforms.Resize(32),
    # transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class FaceDataset(data.Dataset):
    def __init__(self, label_dir, pic_dir, size, transforms=transform):  # transform要有，因为图片需要transform
        super().__init__()
        # self.label_path = label_path

        self.dataset = []

        self.transforms = transforms
        self.picdict = {}
        # self.labeldict = {}
        i = 0
        for picdirs in os.listdir(pic_dir):
            # if str(size) in picdirs:

            if picdirs.startswith(str(size)):
                self.picdict[i] = os.path.join(pic_dir, picdirs)
                i += 1
        # j = 0
        for labelfiles in os.listdir(label_dir):
            if labelfiles.split('.')[0].endswith(str(size)):

                # self.labeldict[j] = os.path.join(label_dir, labelfiles)
                with open(os.path.join(label_dir, labelfiles)) as f:
                    for data_line in f.readlines():
                        datalist = []
                        if data_line[0].isdigit():
                            datalist.extend(data_line.split())
                            self.dataset.append(datalist)
        # print(len(self.dataset))
                # j += 1
        # print(self.picdict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # make data
        # print(type(self.dataset[item][0]))
        pic_filename = self.dataset[item][0]
        # print(pic_filename)
        pic_subdir = self.picdict[int(pic_filename[0])]
        # print(pic_subdir)
        # print(pic_filename)
        pic_file = os.path.join(pic_subdir, pic_filename)
        img = Image.open(pic_file)
        img_data = self.transforms(img)

        # make label
        offset_x1, offset_y1 = self.dataset[item][2], self.dataset[item][3]
        offset_x2, offset_x2 = self.dataset[item][4], self.dataset[item][5]
        conf = self.dataset[item][1]
        # torch不能直接处理str数据，先用numpy转类型
        offset_target = torch.from_numpy(np.array([offset_x1, offset_y1, offset_x2, offset_x2], dtype=np.float32))
        conf_target = torch.from_numpy(np.array([conf], dtype=np.float32))
        # print([offset_x1, offset_y1, offset_x2, offset_x2])
        # offset_target = torch.tensor([offset_x1, offset_y1, offset_x2, offset_x2],dtype=torch.float32)
        # offset_target = torch.Tensor([offset_x1, offset_y1, offset_x2, offset_x2]) #ValueError: too many dimensions 'str'
        # conf_target = torch.tensor([conf],dtype=torch.float32)
        return img_data, conf_target, offset_target


if __name__ == '__main__':
    # 存放标签和图片的文件夹
    label_path_12 = r"../param/final_label_12.txt"
    label_path_24 = r"../param/final_label_24.txt"
    label_path_48 = r"../param/final_label_48.txt"
    pic_path = r"D:\datasets\save_10261_20190725\pic"
    label_path = r"D:\datasets\save_10261_20190725\label"
    # pic_path_12 = r"D:\datasets\save_10261_20190725\pic\12"
    # pic_path_24 = r"D:\datasets\save_10261_20190725\pic\24"
    # pic_path_48 = r"D:\datasets\save_10261_20190725\pic\48"
    dataset_ = FaceDataset(label_path, pic_path, 48)
    dataloader = data.DataLoader(dataset_, batch_size=256, shuffle=True, num_workers=1, drop_last=True)
    print(len(dataloader))
    for i, (img_data_, confidence_, offset_) in enumerate(dataloader):
        print(img_data_.size())
        print(confidence_.size())
        print(offset_.size())
        if i>1:
            print(i)
            print(img_data_)
            break
    #     '''
    #     torch.Size([512, 3, 12, 12])
    #     torch.Size([512, 1])
    #     torch.Size([512, 4])
    #     '''
    #     torch.set_printoptions(threshold=np.inf)
    #     # 第一种算法
    #     confidence_mask = torch.lt(confidence_, 2).view(-1)
    #     offset_mask = torch.gt(confidence_, 0).view(-1)
    #     confidence = confidence_[confidence_mask]
    #     offset = offset_[offset_mask]
    #
    #     # 第二种算法
    #     confidence_mask_1 = torch.lt(confidence_, 2)
    #     confidence_index_1 = torch.nonzero(confidence_mask_1[:, 0])
    #     confidence_1 = confidence_[confidence_index_1]
    #     offset_mask_1 = torch.gt(confidence_, 0)
    #     offset_index_1 = torch.nonzero(offset_mask_1)[:, 0]
    #     offset_1 = offset_[offset_index_1]
    #
    #     # print(confidence)
    #     # print(offset)
    #     print(offset_.size(), offset.size(), offset_1.size())
    #     print(confidence_.size(), confidence.size(), confidence_index_1.size())
    #
    #     # print(offset_mask)
    #     # print(confidence_mask.size())
    #     # confidence = torch.masked_select()
    #
    #     break
    #
    # # for i,(input,confidence_target,offset_target) in enumerate(dataset_):
    # #     # pass
    # #     if i>1000:
    # #         print("input",input.size())
    # #         print("confidence_target",confidence_target)
    # #         print("offset_target",offset_target)
    # #         offset_mask = torch.gt(offset_target, 0)
    # #         print(offset_mask)
    # #     if(i>2000):
    # #         break
    # print(Timekeeper.gettime())
