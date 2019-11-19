import os
import json
# import numpy as np
import shutil

'''
    数据格式：根目录+check目录
    程序提供：根目录和保存文件名
    v1_2主要改动，将定义文件夹路径放在外面，save文件夹不删除
    v1_3主要改动，将图片大小信息加入
    v1_4主要改动，改变标签输出，加入标题行，并对齐
    v1_5主要改动，去掉numpy排序部分，改用列表
'''
def makedir(path):
    '''
    如果文件夹不存在，就创建
    :param path:路径
    :return:路径名
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    return path



# 找到json文件读取其中数据
def makeLabel(rootpath, savefile, needcheck=False, needmake=True, preclear=False, removeProblem=False, mode='w'):
    '''
    生成标签文件，将makeLable和checkdataset两个函数联用
    :param rootpath:根目录
    :param savename:标签保存名
    :param needcheck:是否需要对源数据进行检查，默认False
    :param preclear:是否需要预先对datasets文件夹和save文件夹进行清空，默认False
    :param removeProblem:是否需要移除不良数据，默认False
    :param mode:文件读写模式，默认'w'，备用参数
    :return:data的数量
    步骤：
        1.执行检查步骤，判断要生成标签的文件夹datasets是否存在，返回布尔值和json和jpg路径
        2.datasets存在，则生成保存的文件夹，生成数据列表，遍历json文件夹，读取数据并将结果按格式加入列表
        3.将列表转为数组，按文件名由小到大排序，并将结果保存，并返回数据个数
    '''

    # json文件
    flag, jsonpath, picpath = checkdataset(rootpath, needcheck, preclear)

    if flag and needmake:
        # savedirpath = os.path.join(rootpath, 'save')
        # savepath = os.path.join(savedirpath, savename)
        # makedir(savedirpath)
        jsonfiles = os.listdir(jsonpath)
        datalist = []

        for file in jsonfiles:
            filepath = os.path.join(jsonpath, file)

            with open(filepath, 'r', encoding='utf-8') as load_f:
                try:
                    load_dict = json.load(load_f)
                    xy = load_dict['outputs']['object'][0]['bndbox']
                    x, y = xy['xmin'], xy['ymin']
                    x1, y1 = xy['xmax'], xy['ymax']
                    w, h = (x1 - x), (y1 - y)
                    width,height = load_dict['size']['width'],load_dict['size']['height']
                    # 可以根据情况修改
                    jsondata = "{0} {1} {2} {3} {4} {5} {6} {7} {8}".format(file.split('.')[0],
                                                                            x, y, x1, y1, w, h, width,height)
                    datalist.append(jsondata)
                except:
                    '''如果json文件加载不正确，则输出错误文件名，并根据布尔值删除对应的jpg文件
                    '''
                    print("the problem file is :", file)
                    picname = "{0}.jpg".format(file.split(".")[0])
                    abspicname = os.path.join(picpath, picname)
                    # 如果打开不成功，就把对应的原图删除
                    if (removeProblem):
                        os.remove(filepath)
                        if (os.path.exists(abspicname)):
                            os.remove(abspicname)
                            print(abspicname)
                    continue
        datalist = sorted(datalist)

        with open(savefile, mode) as f:
            title = "%-10s %4s %4s %4s %4s %4s %4s %6s %6s" % ("name",
                                                               "x1", "y1", "x2", "y2", "w", "h", "width", "height")
            print(title,file=f)
            print(title)
            # f.writelines(title+'\n')
            # f.write(title+'\n')
            for data_line in datalist:
                data = data_line.strip().split()
                output_str = "%s.jpg %4s %4s %4s %4s %4s %4s %6s %6s"%tuple([d for d in data])
                print(output_str)
                print(output_str,file=f)
                # f.writelines(output_str+'\n')
                # f.write(output_str + '\n')

        return len(datalist)
    return


# 检查json文件和jpg文件的一一对应性，将具有一一对应的文件复制datasets文件夹
def checkdataset(rootpath, needcheck=False, preclear=False):
    '''

    :param rootpath:根目录
    :param needcheck:是否需要对源数据进行检查，默认False
    :param preclear:是否需要预先对datasets文件夹和save文件夹进行清空，默认False
    :return:布尔值和json和jpg路径
    步骤
        1.定义目标文件夹：
            checkpath   源数据路径
            datasetpath 要保存的路径
            jsonpath    json文件保存路径
            picpath     pic文件保存路径
        2.判断是否需要进行检查，如不需要，而目标预处理数据文件在则返回True,否则返回False,不进行makelabel操作
        3.判断是否需要清空原datasets和save文件夹中的旧数据，并根据路径造出相应文件夹
        4.遍历check文件夹，分加获取json和jpg文件，并将不带扩展名的文件名添加到相应列表
        5.再次遍历check文件夹，将其中一一对应的json和jpgy文件复制到datasets中相应的文件夹，如果文件已存在,则不复制

    '''
    # 定义目标文件夹
    # checkpath = os.path.join(rootpath, 'check')
    # datasetpath = os.path.join(rootpath, 'datasets')
    # jsonpath = os.path.join(datasetpath, 'json')
    # picpath = os.path.join(datasetpath, 'jpg')
    datasetpath = os.path.join(rootpath, datasetdir)
    checkpath = os.path.join(rootpath, checkdir)
    jsonpath = os.path.join(datasetpath, jsondir)
    picpath = os.path.join(datasetpath, picdir)
    makedir(datasetpath)
    # makedir(checkpath)
    makedir(jsonpath)
    makedir(picpath)

    if needcheck:
        if preclear:
            delsuccess = precleardataset(rootpath)
            print("delete successful:", delsuccess)
        # 建立目标文件夹
        # makedir(datasetpath)
        # makedir(picpath)
        # makedir(jsonpath)
        jsonlist = []
        piclist = []

        if os.path.exists(checkpath):
            for root, _, files in os.walk(checkpath):
                for file in files:
                    if file.endswith(".jpg"):
                        piclist.append(file.split('.')[0])
                    if file.endswith(".json"):
                        jsonlist.append(file.split('.')[0])
            for root, _, files in os.walk(checkpath):
                for file in files:
                    if file.endswith(".jpg"):
                        name = file.split('.')[0]
                        if name in jsonlist:
                            pic_path_from = os.path.join(root, file)
                            pic_path_to = os.path.join(picpath, file)
                            try:
                                if not os.path.exists(pic_path_to):
                                    shutil.copy(pic_path_from, pic_path_to)
                            except:
                                raise IOError
                                print("can not copy")
                    if file.endswith(".json"):
                        name = file.split('.')[0]
                        if name in piclist:
                            json_path_from = os.path.join(root, file)
                            json_path_to = os.path.join(jsonpath, file)
                            try:
                                if not os.path.exists(json_path_to):
                                    shutil.copy(json_path_from, json_path_to)
                            except:
                                raise IOError
                                print("can not copy")
            return True, jsonpath, picpath
    elif os.path.exists(picpath) and os.path.exists(jsonpath):
        return True, jsonpath, picpath
    return False, jsonpath, picpath





# 删除文件夹和文件夹下所有内容
def del_file(path, deletedir=False):
    '''
    删除文件夹中的所有内容，并删除本文件夹（文件可以删除，但文件夹经常不能成功删除）
    :param path:要删除的文件路径
    :param deleteDir:是否删除文件夹
    :return:是否成功删除文件
    '''
    flag = False
    if os.path.exists(path):
        if os.path.isdir(path):
            if not os.listdir(path):
                if deletedir:
                    os.rmdir(path)
                    flag = True
                    print(flag)
            else:
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path, deletedir)
                    else:
                        os.remove(c_path)
                        flag = True
        elif os.path.isfile():
            os.remove(path)
            flag = True
    return flag


def precleardataset(rootpath):
    '''
    预先清空datasets and save文件夹
    :param rootpath:根目录
    :return:是否成功删除
    '''
    datasetpath = os.path.join(rootpath, datasetdir)
    # savedirpath = os.path.join(rootpath, 'save')

    def cleardir(path):
        if os.path.isdir(path):
            del_file(path)
            return True
        else:
            return False

    return cleardir(datasetpath)
    # return cleardir(datasetpath), cleardir(savedirpath)

rootpath = r"D:\datasets"
savename = "label_check2.txt"#文件保存名
checkdir = "check2"#原数据文件夹

datasetdir = "datasets2"
jsondir = "json"#dataset中的json文件夹
picdir = "jpg"#dataset中的jpg文件夹

savedir = "save"

savepath = os.path.join(rootpath, savedir)
savefile = os.path.join(savepath,savename)

makedir(savepath)
if __name__ == '__main__':
    num = makeLabel(rootpath, savefile, needcheck=True,needmake=True, preclear=False, removeProblem=False)
    print("The number of data is ", num)
