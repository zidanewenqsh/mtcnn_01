from tool import *



def area(box):
    return torch.mul((box[2] - box[0]), (box[3] - box[1]))


def areas(boxes):
    return torch.mul((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))


def makedir(path):
    '''
    如果文件夹不存在，就创建
    :param path:路径
    :return:路径名
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def toTensor(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, torch.FloatTensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, (list, tuple)):
        return torch.tensor(list(data)).float()  # 针对列表和元组，注意避免list里是tensor的情况
    elif isinstance(data, torch.Tensor):
        return data.float()
    return


def toNumpy(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(list(data))  # 针对列表和元组
    return


def toList(data):
    '''

    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, (list, tuple)):
        return list(data)  # 针对列表和元组
    return


def isBox(box):
    '''
    判断是否是box
    :param box:
    :return:
    '''
    box = toNumpy(box)
    if box.ndim == 1 and box.shape == (4,) and np.less(box[0], box[2]) and np.less(box[1], box[3]):
        return True
    return False


def isBoxes(boxes):
    '''
    判断是否是boxes
    :param boxes:
    :return:
    '''
    boxes = toNumpy(boxes)
    if boxes.ndim == 2 and boxes.shape[1] == 4:
        if np.less(boxes[:, 0], boxes[:, 2]).all() and np.less(boxes[:, 1], boxes[:, 3]).all():
            return True
    return False


# iou
def iou(box, boxes, isMin=False):
    '''

    :param box:
    :param boxes:
    :param isMin:
    :return:
    '''
    '''
    define iou function
    '''

    box = toTensor(box)

    boxes = toTensor(boxes)  # 注意boxes为二维数组

    # 如果boxes为一维，升维
    if boxes.ndimension() == 1:
        boxes = torch.unsqueeze(boxes, dim=0)

    box_area = area(box)
    boxes_area = areas(boxes)
    xx1 = torch.max(box[0], boxes[:, 0])
    yy1 = torch.max(box[1], boxes[:, 1])
    xx2 = torch.min(box[2], boxes[:, 2])
    yy2 = torch.min(box[3], boxes[:, 3])

    inter = torch.mul(torch.max((xx2 - xx1), torch.Tensor([0, ])), torch.max((yy2 - yy1), torch.Tensor([0, ])))

    if (isMin == True):
        over = torch.div(inter, torch.min(box_area, boxes_area))  # intersection divided by union
    else:
        over = torch.div(inter, (box_area + boxes_area - inter))  # intersection divided by union
    return over


def nms(boxes_input, threhold=0.3, isMin=False):
    '''
    define nms function
    :param boxes_input:
    :param isMin:
    :param threhold:
    :return:
    '''

    if isBoxes(boxes_input[:, :4]):
        '''split Tensor'''
        boxes = toTensor(boxes_input)

        boxes = boxes[torch.argsort(-boxes[:, 4])]
        r_box = []
        while (boxes.size(0) > 1):
            r_box.append(boxes[0])
            mask = torch.lt(iou(boxes[0], boxes[1:], isMin), threhold)
            boxes = boxes[1:][mask]  # the other row of Tensor
            '''mask 不能直接放进来,会报IndexError'''
        if (boxes.size(0) > 0):
            r_box.append(boxes[0])
        if r_box:
            return torch.stack(r_box)  # 绝对不能转整数，要不然置信度就变成0
    elif isBox(boxes_input):
        return toTensor(boxes_input)
    return torch.Tensor([])



def offsetToCord(offset, _side):
    x1 = offset[0] * _side
    y1 = offset[1] * _side
    x2 = _side + offset[2] * _side
    y2 = _side + offset[3] * _side
    box = [int(x) for x in [x1, y1, x2, y2]]
    return box



