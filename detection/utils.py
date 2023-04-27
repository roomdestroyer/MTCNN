import numpy as np
import cv2


def filter_boxes(boxes, threshold):
    """执行非最大值抑制（Non-Maximum Suppression，NMS）操作，可以帮助在目标检测中剔除太相似的边界框"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将边界框按照得分值从大到小排序，并记录排序后的下标
    order = scores.argsort()[::-1]

    # 创建一个列表，用于保存最终保留的边界框的下标
    selected_boxes = []

    while order.size > 0:
        # 获取得分值最高的边界框的下标
        i = order[0]
        # 将当前得分值最高的边界框的下标保存到列表中
        selected_boxes.append(i)

        # 获取与当前边界框重叠面积大于阈值的其他边界框的下标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

        # 找出重叠面积小于等于阈值的边界框的下标
        selected_indices = np.where(iou <= threshold)[0]
        # 将这些边界框的下标重新排序，并继续执行下一轮循环
        order = order[selected_indices + 1]

    return selected_boxes


def resize_and_normalize_image(img, scale):
    """将图像按照指定的比率缩放，并将其归一化"""
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    img_resized = (img_resized - 127.5) / 128
    return img_resized


def pad(bboxes, w, h):
    """将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    返回值：
      dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
      edy, edx : n为调整后的box右下角相对原box左上角的相对坐标
      y, x : 调整后的box在原图上左上角的坐标
      ex, ex : 调整后的box在原图上右下角的坐标
      tmph, tmpw: 原始box的长宽
    """
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    """校准box
    参数：
      bbox:pnet生成的box

      reg:rnet生成的box偏移值
    返回值：
      调整后的box是针对原图的绝对坐标
    """

    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


def generate_bbox_from_pnet(cls_map, reg_map, resize_scale, threshold):
    """从输入的分类映射（cls_map）和回归映射（reg_map）中生成边界框（bounding boxes）"""
    # 保留置信度高于阈值的位置
    above_threshold_indices = np.where(cls_map > threshold)
    # 如果没有人脸，返回空数组
    if above_threshold_indices[0].size == 0:
        return np.array([])

    # 获取每个边界框的偏移量
    dx1, dy1, dx2, dy2 = [reg_map[above_threshold_indices[0], above_threshold_indices[1], i] for i in range(4)]
    # 将偏移量组合成一个数组
    reg_offsets = np.array([dx1, dy1, dx2, dy2])
    # 从cls_map中获取得分（置信度）
    score = cls_map[above_threshold_indices[0], above_threshold_indices[1]]

    # 计算在原始图像中的边界框坐标、分类分数和边界框偏移量
    # 由于PNet大致将图像size缩小了2倍，所以在计算边界框坐标时，需要将above_threshold_indices中的行列索引乘以2，才能得到在原始图像中的坐标
    boundingbox = np.vstack([np.round((2 * above_threshold_indices[1]) / resize_scale),
                             np.round((2 * above_threshold_indices[0]) / resize_scale),
                             np.round((2 * above_threshold_indices[1] + 12) / resize_scale),
                             np.round((2 * above_threshold_indices[0] + 12) / resize_scale),
                             score, reg_offsets])
    # 结果数组的形状为 (n, 9)
    return boundingbox.T
