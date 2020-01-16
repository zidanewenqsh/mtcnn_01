import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt
from datasets import faceDatasets

import time


class Trainer():
    def __init__(self, net, save_path, label_path, pic_path, alpha=0.5, save_dir=r"../save/net", isCuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.save_path = save_path
        self.save_dir = save_dir
        self.label_path = label_path
        self.pic_path = pic_path
        self.isCuda = isCuda
        self.alpha = alpha  # 损失的权重
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError

        self.cls_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
        self.net = net
        if os.path.exists(self.save_path):
            print(self.save_path)
            self.net.load_state_dict(torch.load(self.save_path))
            print("load successful")
        # else:
        #     self.net = net
        #     # self.net._paraminit()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.isCuda:
            self.net = self.net.to(self.device)

    def train(self):
        from torch.utils.tensorboard import SummaryWriter
        summaryWriter = SummaryWriter()
        epoch = 0
        start_time = time.time()
        facedataset = faceDatasets.FaceDataset(self.label_path, self.pic_path)
        dataloader = data.DataLoader(facedataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
        # save_file = os.path.join(self.save_path,save_name)
        # losses = []
        while True:
            for i, (img_data_, cls_, offset_) in enumerate(dataloader):
                # start_time_eachbatch = time.time()

                if self.isCuda:
                    img_data_ = img_data_.to(self.device)
                    cls_ = cls_.to(self.device)
                    offset_ = offset_.to(self.device)
                # 输出变形
                _output_cls, _output_offset = self.net(img_data_)
                _output_cls = _output_cls.view(-1, 1)
                _output_offset = _output_offset.view(-1, 4)
                # 分别取标签置信度0,1的数据用来训练置信，和置信度为1,2的数据用来训练偏移量
                # 制作掩码
                cls_mask_ = torch.lt(cls_, 2).view(-1)
                offset_mask_ = torch.gt(cls_, 0).view(-1)
                print("cls_mask_", cls_mask_.size(), torch.lt(cls_, 2).size())
                print("cls_",cls_.size())
                cls_mask = torch.lt(cls_, 2)
                offset_mask = torch.gt(cls_[:, 0], 0)
                print("cls_mask", cls_mask.size())
                # 取标签数据
                cls = cls_[cls_mask]
                offset = offset_[offset_mask]
                # print(_output_cls.size(),cls_mask.size())
                # 取output数据
                output_cls = _output_cls[cls_mask]
                output_offset = _output_offset[offset_mask]

                #第二种算法
                cls_mask_1 = torch.lt(cls_,2)
                print("cls_mask_1", cls_mask_1.size())
                cls_index_1 = torch.nonzero(cls_mask_1)[:,0]
                print("cls_index_1", cls_index_1.size(), torch.nonzero(cls_mask_1).size())
                # print(torch.nonzero(cls_mask_1))
                cls_1 = cls_[cls_index_1]
                print("cls_1", cls_1.size())
                offset_mask_1 = torch.gt(cls_,0)
                print("offset_mask_1",offset_mask_1.size())
                offset_index_1 = torch.nonzero(offset_mask_1)[:,0]
                print("offset_index_1",offset_index_1.size())
                offset_1 = offset_[offset_index_1]
                print("offset_1",offset_1.size())

                # print(cls.size(),output_cls.size())
                # print(offset.size(),output_offset.size())
                # print(output_cls.size(),cls.size())
                # 计算损失
                cls_loss = self.cls_loss(output_cls, cls)
                offset_loss = self.offset_loss(offset, output_offset)
                loss = self.alpha * cls_loss + (1 - self.alpha) * offset_loss
                # acc = torch.mean(torch.lt(torch.abs(torch.div(torch.sub(offset, output_offset), torch.add(offset, 1e-12))),0.2).float())

                # acc = torch.abs(torch.div(torch.sub(offset, output_offset.cpu()), torch.add(offset, 1e-12)))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                summaryWriter.add_scalar("loss", loss, global_step=i)
                # 参数总处理
                params = []
                for param in self.net.parameters():
                    params.extend(param.view(-1).data)
                summaryWriter.add_histogram("params",torch.Tensor(params),global_step=i)
                # 每个参数分别处理
                # for name,param in self.net.named_parameters():
                #         summaryWriter.add_histogram(name,param.data,global_step=i)

                checktime = time.time() - start_time

                if i % 10 == 0:
                    # checktime_2 = time.time() - start_time_eachbatch
                    cls_acc = torch.mean(torch.lt(torch.abs(torch.sub(cls, output_cls)), 0.02).float())
                    offset_acc = torch.mean(torch.lt(torch.abs(torch.sub(offset, output_offset)), 0.02).float())
                    print(
                        "epoch: %d, batch: %d,loss: %f, cls_loss: %f, offset_loss: %f, total_time = %.2fs, cls_acc: %f, offset_acc: %f"
                        % (epoch, i, loss, cls_loss, offset_loss, checktime, cls_acc, offset_acc))
                    # summaryWriter.add_scalar("cls_acc", cls_acc, global_step=i)
                    # summaryWriter.add_scalar("offset_acc", offset_acc, global_step=i)
                    # losses.append(loss.float())
                    # plt.clf()
                    # plt.plot(losses)
                    # plt.pause(0.01)
                    # torch.save(self.net.state_dict(), self.save_path)
                    print("save successful")

            # with torch.no_grad():
            #     self.net.eval()

            save_name = "{0}_{1}.pth".format(self.net.name, epoch)
            save_file = os.path.join(self.save_dir, save_name)
            # print(save_name)
            # print(save_file)
            # torch.save(self.net, save_file)
            print("an epoch save successful")
            epoch += 1
        # end_time = time.time()-start_time
        # print("time",end_time)
