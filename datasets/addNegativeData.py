import os
import numpy as np
import torch
from PIL import Image
from tool import utils

pic_path = r"E:\mtcnndataset\爬虫爬的"
save_path = r"E:\mtcnndataset\save\dataset_gen"
pic_savepath = os.path.join(save_path, "pic")
label_savepath = os.path.join(save_path, "label")
"define iou threshold"
# positive_range = [0.6, 1]
# part_range = [0.2, 0.4]
# negative_range = [0, 0.05]
# size_scale = 0.9
# wh_scale = 0.3
# wh_scales = [0.3, 1, 2]
face_sizes = [12, 24, 48]
classification = ['negative', 'positive', 'part']
# define 每张图片新生成图片的数量
each_pic_num = 11
utils.makedir(pic_savepath)
utils.makedir(label_savepath)
if __name__ == '__main__':
    # 图片数量
    pic_num = 0
    offset = [0.0, 0.0, 0.0, 0.0]
    confidence = 0

    for face_size in face_sizes:
        labellist = []
        # print(len(os.listdir(pic_path)))1844

        for i, data in enumerate(os.listdir(pic_path)):

            if data.endswith(".jpg"):
                pic_absfile = os.path.join(pic_path, data)

                with Image.open(pic_absfile) as img:
                    width, height = img.size
                    if min(width, height) < 80 or img.mode != 'RGB':
                        continue
                    j = 0
                    while j < each_pic_num:
                        # 生成正方形
                        side_len = np.random.randint(40, min(width, height))
                        x0, y0 = width // 2, height // 2
                        # 中心点偏移
                        x0_ = x0 + np.random.randint(-width // 4, width // 4)
                        y0_ = y0 + np.random.randint(-height // 4, height // 4)
                        x1_ = x0_ - side_len // 2
                        y1_ = y0_ - side_len // 2
                        x2_ = x0_ + side_len // 2
                        y2_ = y0_ + side_len // 2
                        if min(x1_, y1_) < 0 or x2_ > width or y2_ > height:
                            continue
                        else:
                            j += 1
                        # 生成偏移量
                        # offset_x1 = (x1 - x1_) / side_len
                        # offset_y1 = (y1 - y1_) / side_len
                        # offset_x2 = (x2 - x2_) / side_len
                        # offset_y2 = (y2 - y2_) / side_len
                        '''
                        负样本不用偏移量'''

                        # 图片保存
                        pic_name = "%06d_%d_1%03d.jpg" % (int(data.strip().split('.')[0][3:]), face_size, j)
                        # print("pic_name",pic_name)
                        pic_savedir = os.path.join(pic_savepath, str(face_size), 'negative')
                        utils.makedir(pic_savedir)
                        save_file = os.path.join(pic_savedir, pic_name)
                        print("save_file", save_file)
                        # 生成标签

                        label = utils.generateLabeldata(pic_name, confidence, offset)
                        # print(label)
                        labellist.append(label)

                        box = (x1_, y1_, x2_, y2_)
                        img_new = img.crop(box)
                        img_new = img_new.resize((face_size, face_size))
                        img_new.save(save_file)
                        # imgTreat()不要用这个，因为图片已经打开
                        pic_num += 1
                        # if j>2:
                        #     print("j",j)
                        #     break
            label_savefile = os.path.join(label_savepath, "negative_{0}.txt".format(face_size))
            with open(label_savefile, 'w') as f:
                for label_data in labellist:
                    print(label_data, file=f)
            # if(i>1):
            #     print("i",i)
            #     break
    print("pic num:", pic_num)
    # print(labellist)
