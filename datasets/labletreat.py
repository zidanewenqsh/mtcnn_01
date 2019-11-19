import os
import time
from tool import utils
'''
处理特殊问题的，可以不用了，里面一些小方法还是可以用的'''
# def generateLabeldata_2(picname, offset, confidence):
#     '''
#     将图片名称，偏移量，和置信处理成一组字符数据
#     改变：输出格式，将confidence提前到第2位
#     :param picname: str
#     :param offset: list
#     :param confidence: int
#     :return:
#     '''
#     if confidence == 0:
#         offset = [0.0, 0.0, 0.0, 0.0]
#     data = []
#     data.append(picname)
#     data.append(confidence)
#     data.extend(offset)
#
#     label = "%s %11f %9f %9f %9f %9f" % tuple([d for d in data])
#     return label

def moveListValue(lt,index_from,index_to):
    '''
    将list中的某一元素移动到另一位置
    :param lt:
    :param index_from:
    :param index_to:
    :return:
    '''
    value_tochange = lt[index_from]
    lt.pop(index_from)
    lt.insert(index_to,value_tochange)
    return lt

def changeListValue(lt,index_a,index_b):
    '''
    交换list中两个元素的位置
    :param lt:
    :param index_a:
    :param index_b:
    :return:
    '''
    lt[index_a],lt[index_b] = lt[index_b],lt[index_a]
    return lt

def moveListValue1(lt, n):
    '''
    将列表中的元素完成特定的向右移动，参数：列表、移动长度 如：[1, 2, 3, 4, 5]，移动2，结果：4, 5, 1, 2, 3
    :param lt:
    :param n:
    :return:
    '''
    # lt = []
    for i in range(n % len(lt)):
        lt.insert(0, lt.pop())
    return lt

def readlabletxt(read_file,datatype='list'):
    '''
    将标签文件是的信息读入并写入字典或列表
    :param read_file:
    :return:
    '''
    a=0
    b=0
    # read_file = r"D:\datasets\save\dataset\label\label_12.txt"
    read_dict = {}
    read_list = []
    # read_set = set()
    with open(read_file) as f:
        read_lines = f.readlines()
        print("lenreadline", len(read_lines))
        if len(read_lines)==0:
            return
        for i, line in enumerate(read_lines):
            if i > 1 or line[0].isdigit():
                data = line.strip().split()
                # print(data[0] not in read_dict)
                a+=1
                if data[0] not in read_dict:
                    read_dict[data[0]] = data[1:]
                    read_list.append(data)
                    b+=1
    # print(a,b)
    if datatype == 'dict':
        return read_dict
    elif datatype == 'list':
        # print("readlist",read_list)
        return read_list
    else:
        raise TypeError

def generateLabelList(read_data):
    '''
    将所有读入的字典或列表返回规定格式的列表
    :param read_dict:
    :return:
    '''
    if len(read_data) == 0:
        return
    datalist = []
    if isinstance(read_data,dict):
    #在这一步做的位置交换
        for key, value in read_data.items():
            # 如果原数据需要修改，在此处修改
            pass
            # picname = key
            # confidence = float(value[4])
            # offset = [float(x) for x in value[0:4]]
            # # print("confidence:",confidence)
            # # print("offset",offset)
            # label = utils.generateLabeldata(picname, offset, confidence)
            # datalist.append(label)
        return datalist
    elif isinstance(read_data,list):
        for value in read_data:
            picname = value[0]
            confidence = float(value[1])
            offset = [float(x) for x in value[2:6]]
            # print("confidence:",confidence)
            # print("offset",offset)
            label = utils.generateLabeldata(picname, confidence, offset)
            datalist.append(label)
    return datalist

def writeLabel(datalist,write_file,mode = 'a'):
    '''
    将内容排序后写入txt文件
    :param datalist:
    :param write_file:
    :param mode:
    :return:
    '''
    with open(write_file,mode) as f:
        for data in sorted(datalist):
            print(data,file=f)
            print("The data is writed successfully")
    return
def counttxtLines(txt_file):
    '''
    计算txt文件里有多少行
    :param txt_file:
    :return:
    '''
    if txt_file.endswith(".txt"):
        with open(txt_file) as f:
            return len(f.readlines())
    return



def mergetxtFile(read_path,write_path,mode='w'):
    '''
    将多个txt文件的内容合并
    :param read_path:
    :param write_path:
    :param mode:
    :return:
    '''
    sizes = [12,24,48]

    #read
    # for root,dirs,files in os.walk(read_path):
    for size in sizes:
        datalist_write = []
        for file in os.listdir(read_path):
            if file.endswith(".txt") and file.split('.')[0].split('_')[-1]==str(size):
                read_file = os.path.join(read_path,file)
                print("readfile",read_file)
                #读取数据，如果datatype为"dict",说明数据需要修改，一般为list
                read_data = readlabletxt(read_file, datatype='list')
                if len(read_data) == 0:
                    continue
                # print(len(readlabletxt(read_file,datatype='list')))
                # datalist_write.extend(readlabletxt(read_file, datatype = 'list'))#AttributeError: 'list' object has no attribute 'strip'
                datalist_write.extend(generateLabelList(read_data))
            else:
                continue
            if len(datalist_write)!=0:
                save_name = "final_label_{0}.txt".format(size)
                write_file = os.path.join(write_path, save_name)
                print("writefile",write_file)
                utils.generateLabel(sorted(datalist_write), write_file, str(size))#sorted是排序并返回列表



if __name__ == '__main__':
    # read_path = r"../param"#直接读文件夹PermissionError: [Errno 13] Permission denied: 'D:\\datasets\\copytest\\5'
    read_path = r"D:\datasets\save_10261_20190723\label_temp"
    # write_path = r"D:\datasets\copytest"
    write_path = r"D:\datasets\save_10261_20190723\label"
    # write_path2 = r"D:\datasets\copytest\4"
    # read_file1 = os.path.join(read_path,"flabel_12.txt")
    # read_file2 = os.path.join(read_path, "flabel_24.txt")
    # read_file3 = os.path.join(read_path, "flabel_48.txt")

    start_time = time.time()

    mergetxtFile(read_path,write_path)
    # _modifyLabel(write_path1,write_path2)
    end_time = time.time()
    time_used = end_time - start_time
    print("time:",time_used)
    # a = readlabletxt(read_file1,datatype='dict')
    # b = readlabletxt(read_file2, datatype='dict')
    # c = readlabletxt(read_file3, datatype='dict')
    # print(len(a),len(b),len(c))




