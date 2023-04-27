import os
import numpy as np
import config
import cv2
import requests
from tqdm import tqdm as Tqdm
import zipfile
npr = np.random


class BBox:
    # 用于表示人脸框
    def __init__(self, box):
        # bbox的左、上、右、下坐标
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]

        # bbox的左上坐标和宽高
        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]

    def map_absolute_to_relative(self, landmark):
        """将每个关键点从绝对坐标系转换为相对坐标系"""
        mapped_landmark = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            point = landmark[i]
            x = (point[0] - self.x) / self.w
            y = (point[1] - self.y) / self.h
            mapped_landmark[i] = np.asarray([x, y])
        return mapped_landmark

    def map_relative_to_absolute(self, landmark):
        """将每个关键点从相对坐标系转换为绝对坐标系"""
        mapped_landmark = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            point = landmark[i]
            x = self.x + self.w * point[0]
            y = self.y + self.h * point[1]
            mapped_landmark[i] = np.asarray([x, y])
        return mapped_landmark


class TestLoader:
    # 制作迭代器
    def __init__(self, image_paths, batch_size=1, shuffle=False):
        # imdb 是一个包含所有图像路径的列表
        # batch_size 是迭代器一次返回的图像数量
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(image_paths)

        self.cur = 0
        self.data = None

        self.reset()
        self.get_batch()

    def __iter__(self):
        """使对象成为迭代器对象，返回自身"""
        return self

    def __next__(self):
        """当迭代器被循环调用时，返回每次迭代的结果"""
        return self.next()

    def reset(self):
        """将迭代器置于起始位置，并初始化数据集"""
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.image_paths)

    def iter_next(self):
        """检查数据集是否可以继续迭代下去"""
        return self.cur + self.batch_size <= self.size

    def next(self):
        """当前批次迭代完毕后，获取下一批数据并返回"""
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def get_batch(self):
        """获取下一批数据"""
        img_path = self.image_paths[self.cur]
        img = cv2.imread(img_path)
        self.data = img


def gen_data_from_lfw():
    """读取LFW数据集中训练集图片的标签信息，将每张图片的路径、人脸框和人脸关键点保存在一个元组中，
    并将所有元组存储在列表result中返回"""
    lfw_train_labels = config.lfw_train_labels
    with open(lfw_train_labels, 'r') as f:
        lines = f.readlines()
    data = []
    # 遍历每一行数据
    for line in lines:
        # 去除字符串(line)首尾的空格或者换行符
        line = line.strip()
        # 按空格分割成多个子串
        components = line.split(' ')
        # 获取图像路径
        img_path = os.path.join(config.LFW_base_dir, components[0]).replace('\\', '/')
        # 构造人脸框
        bbox = (components[1], components[3], components[2], components[4])
        # 将坐标转换为float类型，并取整
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int, bbox))

        # 五个关键点(x,y)
        img_landmark = np.zeros((5, 2))
        # 逐个读取每个关键点的坐标
        for index in range(5):
            img_landmark[index] = (float(components[5 + 2 * index]), float(components[5 + 2 * index + 1]))

        # 将图像路径、人脸框和人脸关键点保存在一个元组中，并添加到结果列表中
        data.append((img_path, BBox(bbox), img_landmark))

    return data


def gen_data_from_wider():
    """从WIDER数据集中读取图像的路径和标注框的坐标，并以字典的形式返回这些信息"""
    label_path = config.wider_face_train_bbx_gt
    f = open(label_path, 'r')
    data = dict()

    # 定义两个空列表分别用于存储图像路径和标注框的坐标
    image_paths = []
    gt_boxes = []

    while True:
        # 图像地址
        img_path = f.readline().strip('\n')
        if not img_path:
            break
        img_path = os.path.join(config.WIDER_train_images_dir, img_path)
        image_paths.append(img_path)

        # 读取当前行的标注框数量
        nums = f.readline().strip('\n')
        if nums == '0':
            _ = f.readline().strip('\n')
        # 定义一个列表用于存储当前图像的所有标注框的坐标
        img_bboxes = []

        for i in range(int(nums)):
            bb_info = f.readline().strip('\n').split(' ')
            # 将标注框的坐标从 [x, y, w, h] 转换为 [xmin, ymin, xmax, ymax] 的形式
            face_box = [float(bb_info[i]) for i in range(4)]
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            img_bboxes.append([xmin, ymin, xmax, ymax])

        gt_boxes.append(img_bboxes)

    # 将 img_paths 和 bboxes 存储到 data 字典中
    data['img_paths'] = image_paths
    data['gt_boxes'] = gt_boxes

    return data


def cal_ious(pred_box, gt_boxes):
    """计算矩形框之间的交并比 Intersection over Union（IoU）"""
    # 计算原矩形框的面积
    pred_area = (pred_box[2] - pred_box[0] + 1) * \
                  (pred_box[3] - pred_box[1] + 1)

    # 计算gt_boxes中每个box的面积
    # gt_boxes是一个数组，每行表示一个矩形框的坐标
    # gt_boxes[:, 0]表示所有矩形框的左上角的x坐标，gt_boxes[:, 2]表示所有矩形框的右下角的x坐标，等等
    gt_areas  = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)

    # 计算pred_box与gt_boxes中每个box最左、最上、最右、最下的坐标
    left_bound_list = np.maximum(pred_box[0], gt_boxes[:, 0])
    top_bound_list = np.maximum(pred_box[1], gt_boxes[:, 1])
    right_bound_list = np.minimum(pred_box[2], gt_boxes[:, 2])
    bottom_bound_list = np.minimum(pred_box[3], gt_boxes[:, 3])

    # 重叠部分长宽
    intersection_widths = np.maximum(0, right_bound_list - left_bound_list + 1)
    intersection_heights = np.maximum(0, bottom_bound_list - top_bound_list + 1)
    
    # 重叠部分面积
    intersection_areas = intersection_widths * intersection_heights
    # 非重叠部分的面积
    union_areas = pred_area + gt_areas - intersection_areas

    # 返回pred_box与每一个gt_box的交并比
    return intersection_areas / (union_areas + 1e-10)


def convert_to_square(box):
    """将给定的矩形框(box)转化为正方形(square_box)，使正方形的边长等于最大边长"""
    square_box = box.copy()
    # 获取矩形框的高度和宽度
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找寻矩形最大边长
    max_length = np.maximum(w, h)

    # 分别调整正方形左上、右下角的横、纵坐标
    square_box[:, 0] = box[:, 0] + w * 0.5 - max_length * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_length * 0.5
    square_box[:, 2] = square_box[:, 0] + max_length - 1
    square_box[:, 3] = square_box[:, 1] + max_length - 1

    return square_box


def pnet_mix_img_lists():
    """对于pnet，将pos,part,neg,landmark四种数据的img_lists混在一起"""
    with open(config.pnet_pos_img_list, 'r') as f:
        pos = f.readlines()
    with open(config.pnet_neg_img_list, 'r') as f:
        neg = f.readlines()
    with open(config.pnet_part_img_list, 'r') as f:
        part = f.readlines()
    with open(config.pnet_landmark_img_list, 'r') as f:
        landmark = f.readlines()

    with open(config.pnet_mixed_img_list, 'w') as f:
        # maxnum_neg用于限制负样本的数量
        maxnum_neg = 750000
        if len(neg) > maxnum_neg:
            neg_samples = npr.choice(len(neg), size=maxnum_neg, replace=True)
        else:
            neg_samples = npr.choice(len(neg), size=len(neg), replace=True)
        # 平衡正样本、部分样本和负样本之间的数量关系

        pos_samples = npr.choice(len(pos), len(neg_samples) // 3, replace=True)
        part_samples = npr.choice(len(part), len(neg_samples) // 3, replace=True)

        print('正样本数量：{} 负样本数量：{} 部分样本数量:{}'.format(len(pos_samples), len(neg_samples), len(part_samples)))

        # 最后，将选中的pos，neg，part和所有landmark样本写入pnet_mixed_img_list文件
        for i in pos_samples:
            f.write(pos[i])
        for i in neg_samples:
            f.write(neg[i])
        for i in part_samples:
            f.write(part[i])
        for i in landmark:
            f.write(i)


def create_data_dir():
    """创建一些必要的文件夹"""
    if not os.path.exists(config.data_base_dir):
        os.mkdir(config.data_base_dir)

    # 下载WIDER数据集和LFW数据集及标签
    download_train_datasets()

    # pnet的数据目录
    pnet_data_dir = config.pnet_data_dir
    if not os.path.exists(pnet_data_dir):
        os.mkdir(pnet_data_dir)
    if not os.path.exists(config.pnet_pos_dir):
        os.mkdir(config.pnet_pos_dir)
    if not os.path.exists(config.pnet_part_dir):
        os.mkdir(config.pnet_part_dir)
    if not os.path.exists(config.pnet_neg_dir):
        os.mkdir(config.pnet_neg_dir)

    # rnet的数据目录
    if not os.path.exists(config.rnet_data_dir):
        os.mkdir(config.rnet_data_dir)

    # onet的数据目录
    if not os.path.exists(config.onet_data_dir):
        os.mkdir(config.onet_data_dir)

    # 日志目录和测试目录
    if not os.path.exists(config.logs_dir):
        os.mkdir(config.logs_dir)
    if not os.path.exists(config.test_base_dir):
        os.mkdir(config.test_base_dir)
    if not os.path.exists(config.input_imgs_dir):
        os.mkdir(config.input_imgs_dir)
    if not os.path.exists(config.output_imgs_dir):
        os.mkdir(config.output_imgs_dir)
    
    print("目录文件和数据集已准备完毕，可以开始生成数据和训练网络")


def download_train_datasets():
    """下载训练数据集及其标签，链接有效期到2024年4月26日"""
    # WIDER数据集的训练集下载链接
    url_wider_train = "https://bucket011.obs.cn-north-4.myhuaweicloud.com:443/WIDER_train.zip?AccessKeyId" \
                      "=KRAAOULAOH6EEQIYJVKD&Expires=1713627643&Signature=Xe6fLQcX/aBbZZLInKm8HuI2b%2BU%3D"
    # WIDER数据集的label下载链接
    url_wider_labels = "https://bucket011.obs.cn-north-4.myhuaweicloud.com:443/wider_face_split.zip?AccessKeyId" \
                       "=KRAAOULAOH6EEQIYJVKD&Expires=1713627963&Signature=nemm8N73EmGWvb1IW2%2BonuKpX1A%3D"
    # LFW数据集及其label下载链接
    url_lfw_train_and_labels = "https://bucket011.obs.cn-north-4.myhuaweicloud.com:443/train.zip?AccessKeyId" \
                               "=KRAAOULAOH6EEQIYJVKD&Expires=1713627982&Signature=TKkkrw9GoJdT10C0WgNUnIDM9ic%3D"
    urls = [url_wider_train, url_wider_labels, url_lfw_train_and_labels]

    WIDER_base_dir = config.WIDER_base_dir
    LFW_base_dir = config.LFW_base_dir
    if os.path.exists(WIDER_base_dir) and os.path.exists(LFW_base_dir):
        print("WIDER数据集和LFW数据集已存在，无需下载")
        return
    else:
        if not os.path.exists(WIDER_base_dir):
            os.mkdir(WIDER_base_dir)
        if not os.path.exists(LFW_base_dir):
            os.mkdir(LFW_base_dir)

    # 下载到指定的压缩文件
    path_wider_train = os.path.join(WIDER_base_dir, "WIDER_train.zip")
    path_wider_labels = os.path.join(WIDER_base_dir, "wider_face_split.zip")
    path_lfw = os.path.join(LFW_base_dir, "LFW.zip")
    paths = [path_wider_train, path_wider_labels, path_lfw]

    for url, path in zip(urls, paths):
        print("正在下载文件 %s" % os.path.basename(path), " 保存路径为 %s " % path)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(path, "wb") as file:
            with Tqdm(total=total_size, unit="B", unit_scale=True, desc='downloading: ' + path, ncols=100) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    progress_bar.update(len(chunk))

        print("数据集下载完成，正在解压缩文件")
        # 解压所下载的压缩文件
        output_folder = os.path.dirname(path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # 打开压缩文件并解压
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(output_folder)
            print(f"解压缩文件 {path} 到 {output_folder} 完成")