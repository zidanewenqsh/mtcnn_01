import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from tool import utils

# 定义路径
# root = r"D:\datasets"

"一级改动"
labelfile = r"D:\datasets\datasets10261\label_10261.txt"  # 读取的标签路径
"二级改动"
picpath = r"D:\datasets\datasets10261\jpg"  # 读取的图片路径
"一级改动"
savepath = r"D:\datasets\save_10261_20190726"
utils.makedir(savepath)
# 图片和标签的保存根目录
pic_savepath = os.path.join(savepath, "pic")
label_savepath = os.path.join(savepath, "label")
"define iou threshold"
positive_range = [0.65, 1]
part_range = [0.4, 0.6]
negative_range = [0, 0.05]
size_scale = 0.9
wh_scales = [0.05, 0.1, 0.2, 0.75, 0.95]
face_sizes = [12, 24, 48]
classification = ['negative', 'positive', 'part']

utils.makedir(pic_savepath)
utils.makedir(label_savepath)

# define 每张图片新生成图片的数量
each_pic_num = 10  # 1--144


def getconfidence(iou_value):
    '''

    :param iou_value:
    :return:
    '''
    if positive_range[0] < iou_value < positive_range[1]:
        confidence = 1
    elif part_range[0] < iou_value < part_range[1]:
        confidence = 2
    elif negative_range[0] < iou_value < negative_range[1]:
        confidence = 0
    else:
        return -1
    return confidence


def classify(iou_value):
    '''
    没用
    :param iou_value:
    :return:
    '''
    # confidence = 0
    if positive_range[0] < iou_value < positive_range[1]:
        confidence = 1
    elif part_range[0] < iou_value < part_range[1]:
        confidence = 2
    elif negative_range[0] < iou_value < negative_range[1]:
        confidence = 0
    else:
        return -1
    return {
        0: "negative",
        1: "positive",
        2: "part"
    }.get(confidence, "error")  # 不能加default=


def generatedataset(pass_used=False, treat_img=True, gen_label=True):
    pic_used_dict = {12: "12.txt", 24: "24.txt", 48: "48.txt"}

    # 新旧标签列表
    datalist_read = []
    # datalist_write = []

    with open(labelfile, 'r') as f:
        for i, line in enumerate(f.readlines()):
            # 第一行为标量行，不读
            if (i > 0):
                datalist_read.append(line.strip())

    ioulist = []
    # 定义三个样本的计数
    positive_num_ = 0
    negative_num_ = 0
    part_num_ = 0

    positive_num = 0
    negative_num = 0
    part_num = 0
    for face_size in face_sizes:
        datalist_write = []  # 注意datalist清空位置，注意计数
        '''
           将已生成的图片存起来，下次不再生成'''
        pic_used_txt = pic_used_dict.get(face_size)
        if pass_used:

            pic_used_list = []

            if os.path.exists(pic_used_txt):
                with open(pic_used_txt) as f:
                    for line in f.readlines():

                        if line[0].isdigit():
                            pic_used_list.append(line.strip().split()[0])
        elif os.path.exists(pic_used_txt):
            os.remove(pic_used_txt)

        "定义三个尺寸的文件保存路径"
        # positive_savepath = os.path.join(pic_savepath, str(face_size), "positive")
        # negative_savepath = os.path.join(pic_savepath, str(face_size), "negative")
        # part_savepath = os.path.join(pic_savepath, str(face_size), "part")
        # pic_savepathlist = [negative_savepath,positive_savepath,part_savepath]
        # 定义图片保存文件夹
        # pic_savedict = {
        #     0: os.path.join(pic_savepath, str(face_size), "negative"),
        #     1: os.path.join(pic_savepath, str(face_size), "positive"),
        #     2: os.path.join(pic_savepath, str(face_size), "part")
        # }
        '''
        定义每个尺寸图片保存文件夹和标签保存文件的dict
        图片要按照置信分类，标签不用'''
        save_dict = {
            "pic": {
                0: os.path.join(pic_savepath, str(face_size), "negative"),
                1: os.path.join(pic_savepath, str(face_size), "positive"),
                2: os.path.join(pic_savepath, str(face_size), "part")
            },
            "label": os.path.join(label_savepath, "label_{0}.txt".format(str(face_size)))
        }
        # positive part negative集中放置，这几个文件夹不用生成
        # for i in [0, 1, 2]:
        #     utils.makedir(save_dict['pic'][i])

        "遍历标签数据"
        for i, strdata in enumerate(datalist_read):
            "分隔每行数据"
            data = strdata.strip().split()
            if pass_used:
                if data[0] in pic_used_list:
                    continue

            # x,y,w,h = map(int,data[1],data[2],data[3],data[4])#这个写法不行
            # x1, y1, w, h, width, height = map(int, data[1:7])#本身可以实现数据的strip()
            # x1, y1, w, h, width, height = map(int, map(str.strip,data[1:7]))#不以逗号分隔了，改为空格
            x1, y1, x2, y2, w, h, width, height = map(int, data[1:9])
            "图片过滤"
            # if (x1 < 0 or y1 < 0 or w < 0 or h < 0 or max(w, h) < 40):
            #     continue
            "判断是否有小于0的元素，框的大小是否比40小，框是否在图片内"
            if any([d < 0 for d in map(int, data[1:9])]) or min(w, h) < 40 or max(w, h) > min(width, height):
                # print("false data: ", data)
                # 输出错误数据
                continue
            '''
            补方框'''
            # "判断框是否为近方框"
            # if w / (w + h) > 0.6 or w / (w + h) < 0.3:
            #     pass
            # else:
            #     continue

            # x1, y1, x2, y2, w, h, width, height = map(int, data[1:9])

            "计算右下角坐标x1,y1和中心点x0,y0"
            # x2, y2 = x1 + w, y1 + h
            # x0, y0 = x1 + w // 2, y1 + h // 2
            for j in range(len(wh_scales)):
                k = 0
                while k < each_pic_num:
                    '''
                    name是加载图片的名称
                    pic_name 是新生成图片的名称'''

                    name, offset, box, box_ = utils.getoffset(data, wh_scale=wh_scales[j], size_scale=size_scale)

                    iou_value = utils.iou(box, box_)

                    # 生成图片文件名
                    pic_name = "%s_%d_0%1d%02d.jpg" % (
                        name.split('.')[0], face_size, j, k)  # 文件名中第二个"_"后的第一个0代表正或部分样本，另外生成的负样本此处为1

                    # 生成图片路径
                    "置信"
                    confidence = getconfidence(iou_value)

                    "图片保存路径"
                    if confidence == -1:
                        continue
                    else:
                        ioulist.append(iou_value)
                        k += 1

                    pic_savedir = os.path.join(pic_savepath, str(face_size))
                    utils.makedir(pic_savedir)
                    pic_savefile = os.path.join(pic_savedir, pic_name)

                    "标签保存路径"
                    label_savefile = os.path.join(save_dict['label'])

                    "原图片路径"
                    pic_file = os.path.join(picpath, name)

                    if treat_img:
                        utils.imgTreat(pic_file, pic_savefile, box_, face_size)

                    datalist_write.append(utils.generateLabeldata(pic_name, confidence, offset))
                    # 三个样本计数

                    if positive_range[0] < iou_value < positive_range[1]:
                        positive_num_ += 1
                    elif part_range[0] < iou_value < part_range[1]:
                        part_num_ += 1
                    elif negative_range[0] < iou_value < negative_range[1]:
                        negative_num_ += 1
                if i%100 == 0:
                    total_num_ = positive_num_ + part_num_ + negative_num_
                    print("epoch", i, positive_num_, part_num_, negative_num_, total_num_)
                    if all([positive_num_ > 0, part_num_ > 0, negative_num_ > 0, total_num_ > 0]):
                        print("positive: %.3f, %.3f" % (positive_num_ / part_num_, positive_num_ / negative_num_))
                        print("ratio: %.3f, %.3f, %.3f" % (positive_num_ / total_num_, part_num_ / total_num_,
                              negative_num_ / total_num_))  # 临时查看一下数量
                    print("********************")

            if pass_used:
                with open(pic_used_txt, 'a') as pic_used_file:
                    print(data[0], file=pic_used_file)


        if len(datalist_write) != 0 and gen_label:
            utils.generateLabel(datalist_write, label_savefile, face_size)

    # 三个样本计数
    for i in ioulist:
        if positive_range[0] < i < positive_range[1]:
            positive_num += 1
        elif part_range[0] < i < part_range[1]:
            part_num += 1
        elif negative_range[0] < i < negative_range[1]:
            negative_num += 1
    total_num = positive_num + part_num + negative_num
    print(positive_num, part_num, negative_num, total_num)


if __name__ == '__main__':
    # 清空一下save文件夹
    # preclear(savepath)
    generatedataset(pass_used=False, gen_label=False, treat_img=False)
    # generatedataset(pass_used=False, gen_label=True, treat_img=True)
    # generatedataset(pass_used=True, gen_label=True, treat_img=True)
    print(1)
