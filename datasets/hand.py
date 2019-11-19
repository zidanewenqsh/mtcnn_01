import os
from PIL import Image

pic_dir = r"C:\Users\Administrator\Pictures\手"
pic_savedir = r"C:\Users\Administrator\Pictures\hand"
label_savedir = r"C:\Users\Administrator\Pictures\hand\label"
img_sizes = [12,24,48]
# data_ = [0.0, 0.0, 0.0, 0.0, 0.0]
for img_size in img_sizes:
    datalist = []
    label_name = "handlabel_{0}.txt".format(img_size)
    label_file = os.path.join(label_savedir,label_name)
    print(label_file)
    for i, pic_name in enumerate(os.listdir(pic_dir)):
        pic_file = os.path.join(pic_dir,pic_name)
    # print(pic_file)


        with Image.open(pic_file) as img:
            img = img.resize((img_size,img_size))
            new_name = "200000_%d_0%02d.jpg" %(img_size,i)
            new_savefile = os.path.join(pic_savedir,str(img_size),new_name)
            img.save(new_savefile)
            label_data = "%-18s %10f %9f %9f %9f %9f" %(new_name,0,0,0,0,0)
            # print(new_savefile)
            datalist.append(label_data)
            # print(label_data)
    with open(label_file,'w') as f:
        for data in datalist:
            print(data,file=f)