import os
import random
import sys
import cv2
import numpy as np
from tqdm import tqdm
from preprocess.utils import cal_ious, gen_data_from_lfw, BBox
import config

npr = np.random


def gen_landmark_data(img_size):
    """ 用于处理带有landmark的数据 """

    if img_size == 12:
        # 图片处理后输出路径
        landmark_dir = config.pnet_landmark_dir
        if not os.path.exists(landmark_dir):
            os.mkdir(landmark_dir)
        f = open(config.pnet_landmark_img_list, 'w')
    elif img_size == 24:
        # 图片处理后输出路径
        landmark_dir = config.rnet_landmark_dir
        if not os.path.exists(landmark_dir):
            os.mkdir(landmark_dir)
        f = open(config.rnet_landmark_img_list, 'w')
    elif img_size == 48:
        # 图片处理后输出路径
        landmark_dir = config.onet_landmark_dir
        if not os.path.exists(landmark_dir):
            os.mkdir(landmark_dir)
        f = open(config.onet_landmark_img_list, 'w')
    else:
        print("Invalid image size")
        sys.exit(1)

    # data格式：((img_path, BBox(), img_landmark))
    data = gen_data_from_lfw()

    image_id = 0
    # 遍历每一张图片
    for (img_path, bbox, img_landmark) in tqdm(data):
        # 读取图片
        img = cv2.imread(img_path)
        # 存储人脸图片和关键点
        all_faces, all_landmarks = [], []
        # 图片的宽高和通道
        img_h, img_w, img_c = img.shape
        # 人脸框的坐标
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])

        # 1. 首先将真实的人脸区域和关键点坐标加入到总列表中
        # 裁剪人脸图片区域，+1是为了保证包含边界像素
        face_area = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
        # resize成网络输入大小
        face_area = cv2.resize(face_area, (img_size, img_size))
        # 此时图片的shape为：(img_size, img_size, 3)
        face_landmark = np.zeros((5, 2))
        # 计算五个关键点相对于bbox左上坐标的偏移量并归一化
        for index, one_gt_landmark in enumerate(img_landmark):
            landmark_offset = ((one_gt_landmark[0] - gt_box[0]) / (gt_box[2] - gt_box[0]),
                               (one_gt_landmark[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            face_landmark[index] = landmark_offset
        # 将裁剪后的子图像添加到总列表中
        all_faces.append(face_area)
        # 五个关键点reshape成5*2的一维数组，关键点的表示是相对bbox左上角的偏移量
        all_landmarks.append(face_landmark.reshape(10))

        # 2. 然后执行图像增强，将变换后的人脸区域和关键点坐标加入到总列表中
        # 是否对图像变换
        argument = True
        if argument:
            image_id += 1
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            gt_w = gt_x2 - gt_x1 + 1
            gt_h = gt_y2 - gt_y1 + 1
            # 除去过小图像
            if max(gt_w, gt_h) < 40 or gt_x1 < 0 or gt_y1 < 0:
                continue
            # 为了让模型具有更好的泛化能力，需要从原始的bbox生成一些扰动的边界框，每张图像做10次变换
            for i in range(10):
                # 根据扰动后的边界框，从原始图像中裁剪出对应的人脸区域，这些裁剪出的窗口将作为正样本用于训练网络
                # 随机伸缩图像，伸缩后的矩形框大小介于原始矩形框的最小边长的80%和最大边长的125%之间
                new_box_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                # 分别计算随机平移后的矩形框左上角x坐标和y坐标的偏移量，偏移量的范围分别在原始矩形框宽度和高度的±20%之间
                offset_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                offset_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                # 计算变换后的bbox的左上角坐标
                new_x1 = int(max(gt_x1 + gt_w / 2 - new_box_size / 2 + offset_x, 0))
                new_y1 = int(max(gt_y1 + gt_h / 2 - new_box_size / 2 + offset_y, 0))
                new_x2 = new_x1 + new_box_size
                new_y2 = new_y1 + new_box_size
                # 除去超过边界的图像
                if new_x2 > img_w or new_y2 > img_h:
                    continue
                # crop_box是新的裁剪框
                crop_box = np.array([new_x1, new_y1, new_x2, new_y2])
                # img是原始图像，cropped_img是裁剪后的新图像
                cropped_img = img[new_y1:new_y2 + 1, new_x1:new_x2 + 1, :]
                # 将裁剪后的图像resize成输入输入要求的形状
                resized_img = cv2.resize(cropped_img, (img_size, img_size))

                # iou>0.65的框才可以看作有效框
                iou = cal_ious(crop_box, np.expand_dims(gt_box, 0))
                if iou > 0.65:
                    all_faces.append(resized_img)
                    # 五个关键点相对于new_box的偏移比例
                    face_landmarks = np.zeros((5, 2))
                    for index, one_gt_landmark in enumerate(img_landmark):
                        new_landmark_offset = ((one_gt_landmark[0] - new_x1) / new_box_size,
                                               (one_gt_landmark[1] - new_y1) / new_box_size)
                        face_landmarks[index] = new_landmark_offset
                    all_landmarks.append(face_landmarks.reshape(10))
                    bbox = BBox([new_x1, new_y1, new_x2, new_y2])
                    # 对正样本进行数据增强，如镜像和旋转
                    # 镜像
                    if random.choice([0, 1]) == 1:
                        face_flipped_by_x, face_landmarks_flipped_by_x = flip_by_x(resized_img, face_landmarks)
                        face_flipped_by_x = cv2.resize(face_flipped_by_x, (img_size, img_size))
                        all_faces.append(face_flipped_by_x)
                        all_landmarks.append(face_landmarks_flipped_by_x.reshape(10))
                    # 逆时针旋转
                    if random.choice([0, 1]) == 1:
                        # reprojectLandmark将关键点相对于原点（bbox左上角）的偏移比例变换为绝对坐标
                        face_rotated_by_alpha, face_landmarks_rotated_by_alpha \
                            = rotate(img, bbox, bbox.map_relative_to_absolute(face_landmarks), 5)
                        # projectLandmark将关键点的绝对坐标变换为相对于原点（bbox左上角）的偏移比例
                        face_landmarks_rotated_by_alpha = bbox.map_absolute_to_relative(face_landmarks_rotated_by_alpha)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
                        all_faces.append(face_rotated_by_alpha)
                        all_landmarks.append(face_landmarks_rotated_by_alpha.reshape(10))
                        # 旋转之后再左右翻转
                        face_flipped_by_x, face_landmarks_flipped_by_x \
                            = flip_by_x(face_rotated_by_alpha, face_landmarks_rotated_by_alpha)
                        face_flipped_by_x = cv2.resize(face_flipped_by_x, (img_size, img_size))
                        all_faces.append(face_flipped_by_x)
                        all_landmarks.append(face_landmarks_flipped_by_x.reshape(10))
                    # 顺时针旋转
                    if random.choice([0, 1]) == 1:
                        face_rotated_by_alpha, face_landmarks_rotated_by_alpha \
                            = rotate(img, bbox, bbox.map_absolute_to_relative(face_landmarks), -5)
                        face_landmarks_rotated_by_alpha = bbox.map_relative_to_absolute(face_landmarks_rotated_by_alpha)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (img_size, img_size))
                        all_faces.append(face_rotated_by_alpha)
                        all_landmarks.append(face_landmarks_rotated_by_alpha.reshape(10))
                        # 旋转之后再左右翻转
                        face_flipped_by_x, face_landmarks_flipped_by_x \
                            = flip_by_x(face_rotated_by_alpha, face_landmarks_rotated_by_alpha)
                        face_flipped_by_x = cv2.resize(face_flipped_by_x, (img_size, img_size))
                        all_faces.append(face_flipped_by_x)
                        all_landmarks.append(face_landmarks_flipped_by_x.reshape(10))

        # 将得到的图片列表保存到结果文件
        all_faces, all_landmarks = np.asarray(all_faces), np.asarray(all_landmarks)
        # 剔除F_landmarks中偏移量不在[0, 1]之内的图片
        for i in range(len(all_faces)):
            # 由于在应用随机扰动后，有可能产生超出图像边界的landmark，因此需要将所有landmark限制在图像边界之内
            # 如果F_landmarks[i]中存在一个或多个小于等于0的元素，则返回True，否则返回False
            if np.sum(np.where(all_landmarks[i] <= 0, 1, 0)) > 0:
                continue
            # 如果F_landmarks[i]中存在一个或多个大于等于1的元素，则返回True，否则返回False
            if np.sum(np.where(all_landmarks[i] >= 1, 1, 0)) > 0:
                continue
            cv2.imwrite(os.path.join(landmark_dir, '%d.jpg' % image_id), all_faces[i])
            landmarks = list(map(str, list(all_landmarks[i])))
            f.write(os.path.join(landmark_dir, '%d.jpg' % image_id) + ' -2 ' + ' '.join(landmarks) + '\n')
            image_id += 1

    f.close()


def flip_by_x(face, face_landmarks):
    """ 水平翻转 """
    # 参数1表示进行水平翻转
    face_flipped_by_x = cv2.flip(face, 1)
    # 对face_landmarks进行水平翻转
    landmark_flipped_by_x = np.asarray([(1 - x, y) for (x, y) in face_landmarks])
    # 翻转后左眼变右眼，左嘴角变右嘴角
    landmark_flipped_by_x[[0, 1]] = landmark_flipped_by_x[[1, 0]]
    landmark_flipped_by_x[[3, 4]] = landmark_flipped_by_x[[4, 3]]
    return face_flipped_by_x, landmark_flipped_by_x


def rotate(img, bbox, face_landmarks, alpha):
    """ 旋转 """
    # 计算中心点坐标
    center = ((bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2)
    # alpha表示旋转的角度，参数1表示图像的缩放比例，返回一个二维旋转矩阵
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    # 对原始图像进行旋转变换
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    # 对关键点进行旋转变换
    face_landmarks_by_alpha = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                                          rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])
                                         for (x, y) in face_landmarks])
    face_rotated_by_alpha = img_rotated_by_alpha[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
    return face_rotated_by_alpha, face_landmarks_by_alpha
