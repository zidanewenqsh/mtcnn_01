# from src import *
from tool.utils import *

pic_path = r"D:\datasets\save_10261_20190725\negative_tomake_20190909"
save_path = r"D:\datasets\save_10261_20190725\negavive"
pic_saveabspath = os.path.join(save_path, "pic")
label_savedir = os.path.join(save_path, "label")
pic_savedir_12 = os.path.join(pic_saveabspath, "12")
pic_savedir_24 = os.path.join(pic_saveabspath, "24")
pic_savedir_48 = os.path.join(pic_saveabspath, "48")
label_12 = os.path.join(label_savedir, "nlabel_12.txt")
label_24 = os.path.join(label_savedir, "nlabel_24.txt")
label_48 = os.path.join(label_savedir, "nlabel_48.txt")
pic_savedict = {12: pic_savedir_12, 24: pic_savedir_24, 48: pic_savedir_48}
label_savedict = {12: label_12, 24: label_24, 48: label_48}
makedir(pic_saveabspath)
makedir(label_savedir)
if __name__ == '__main__':
    # 图片数量
    pic_num = 0
    offset = [0.0, 0.0, 0.0, 0.0]
    conf = 0
    sizes = [12, 24, 48]
    ws = []
    hs = []

    count = {12: 0, 24: 0, 48: 0}
    # size = 48
    for size in sizes:
        i = 0
        labellist = []
        for pic_name in os.listdir(pic_path):
            each_num = 0
            pic_file = os.path.join(pic_path, pic_name)

            with Image.open(pic_file) as img_:
                width, height = img_.size
                ws.append(width)
                hs.append(height)
                h = size
                while h <= height - size:
                    w = size
                    while w <= width - size:
                        if count[size] >= 10000:
                            break

                        box = (w, h, w + size, h + size)
                        img = img_.crop(box)
                        pic_newname = "1%05d_%2d_0000.jpg" % (i, size)
                        pic_newpath = os.path.join(pic_savedict[size], pic_newname)
                        # img.save(pic_newpath)
                        print(pic_newpath)

                        label_data = [pic_newname, 0.0, 0.0, 0.0, 0.0, 0.0]
                        label_str = "%-18s %10f %9f %9f %9f %9f" % tuple([d for d in label_data])
                        labellist.append(label_str)
                        # print(pic_newname)
                        # print(label)
                        # print(len(pic_newname))
                        w += 48
                        i += 1
                        pic_num += 1
                        count[size] += 1
                    h += 48
        with open(label_savedict[size],'w') as file:
            for line_data in labellist:
                # print(line_data,file=file)
                pass

    print("pic_num", pic_num)
    print(np.sum(ws) / 48)
    print(np.sum(hs) / 48)
    for k, v in count.items():
        print(k)
        print(v)
