import os
import cv2
import sys
import numpy as np
import numpy.random as npr
from tqdm import tqdm
from preprocess.utils import cal_ious
import config
from preprocess.utils import TestLoader, convert_to_square, gen_data_from_wider
import pickle
from detection.main_detector import MainDetector
from detection.sub_detector import SubDetector
from preprocess.utils import gen_data_from_wider


def gen_source_ppn_samples():
    """使用WIDER数据集生成正样本、负样本、部分样本作为PNet的输入"""

    data = gen_data_from_wider()
    num_images = len(data['img_paths'])
    print('WIDER训练集中的总图片数： %d' % num_images)

    f_pos = open(config.pnet_pos_img_list, 'w')
    f_neg = open(config.pnet_neg_img_list, 'w')
    f_part = open(config.pnet_part_img_list, 'w')

    # 记录pos,neg,part三类生成数
    id_positive = 0
    id_negative = 0
    id_part = 0
    # 记录读取图片的总数
    img_done = 0

    # 遍历每一张图像
    for img_id in tqdm(range(num_images)):

        # 获取每一张图像及其真实标注框
        img_path = data['img_paths'][img_id]
        gt_boxes = np.array(data['gt_boxes'][img_id], dtype=np.float32).reshape(-1, 4)
        if gt_boxes.shape[0] == 0:
            continue
        
        # 读取该图像
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        img_done += 1
        num_neg = 0
        # 随机采样一定的negative样本
        while num_neg <= 50:
            # 随机选取一个边长为[12, 图像宽高中的较小值/2)的正方形，正方形的面积一定小于图像的四分之一
            size = npr.randint(12, min(width, height) / 2)
            # 随机选取左上坐标，并确保以该点为左上角的正方形一定在图像内
            nx1 = npr.randint(0, width - size)
            ny1 = npr.randint(0, height - size)

            # 构造正方形box
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            # 计算该box与别的gt_boxes的iou值
            ious = cal_ious(crop_box, gt_boxes)
            # 用构造出来的box截取图片并使用线性插值resize成12x12大小
            cropped_img = img[ny1:ny1 + size, nx1:nx1 + size, :]
            resized_img = cv2.resize(cropped_img, (12, 12), interpolation=cv2.INTER_LINEAR)
            # iou值小于0.3判定为neg样本
            if np.max(ious) < 0.3:
                save_file = os.path.join(config.pnet_neg_dir, '%s.jpg' % id_negative)
                f_neg.write(config.pnet_neg_dir + '/%s.jpg' % id_negative + ' 0\n')
                cv2.imwrite(save_file, resized_img)
                id_negative += 1
                num_neg += 1

        # 对每个人脸框进行变换，以获得pos、part和neg样本
        for gt_box in gt_boxes:
            # 获取人脸框的左上右下坐标
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            gt_w = gt_x2 - gt_x1 + 1
            gt_h = gt_y2 - gt_y1 + 1
            # 舍去图像过小和box在图片外的图像
            if max(gt_w, gt_h) < 20 or gt_x1 < 0 or gt_y1 < 0:
                continue

            # 对每个人脸框做5次尺度较大的变换，以获取neg样本
            for i in range(5):
                # 随机选取一个边长为[12, 图像宽高中的较小值/2)的正方形，正方形的面积一定小于图像的四分之一
                size = npr.randint(12, min(width, height) / 2)
                # 随机生成的关于x1,y1的偏移量，并且保证x1+delta_x>0,y1+delta_y>0
                delta_x = npr.randint(max(-size, -gt_x1), gt_w)
                delta_y = npr.randint(max(-size, -gt_y1), gt_h)
                # 截取后的左上角坐标
                nx1 = int(max(0, gt_x1 + delta_x))
                ny1 = int(max(0, gt_y1 + delta_y))
                # 排除大于图片尺度的
                if nx1 + size > width or ny1 + size > height:
                    continue

                # 构造正方形box
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                # 计算该box与别的gt_boxes的iou值
                ious = cal_ious(crop_box, gt_boxes)
                # 用构造出来的box截取图片并使用线性插值resize成12x12大小
                cropped_img = img[ny1:ny1 + size, nx1:nx1 + size, :]
                resized_img = cv2.resize(cropped_img, (12, 12), interpolation=cv2.INTER_LINEAR)
                # iou值小于0.3判定为neg图像
                if np.max(ious) < 0.3:
                    save_file = os.path.join(config.pnet_neg_dir, '%s.jpg' % id_negative)
                    f_neg.write(config.pnet_neg_dir + '/%s.jpg' % id_negative + ' 0\n')
                    cv2.imwrite(save_file, resized_img)
                    id_negative += 1

            # 每个人脸框做20次尺度较小的变换，以获得pos和part样本
            for i in range(20):
                # 除去尺度小的图片
                if gt_w < 5:
                    continue
                # 缩小随机选取size范围，更多截取pos和part图像
                size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(max(gt_w, gt_h) * 1.25))
                # 偏移量，范围缩小了
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                # 先计算x1+w/2-size/2即左上角坐标，再+delta_x偏移量
                nx1 = int(max(gt_x1 + gt_w / 2 - size / 2 + delta_x, 0))
                ny1 = int(max(gt_y1 + gt_h / 2 - size / 2 + delta_y, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                # 排除超出的图像
                if nx2 > width or ny2 > height:
                    continue
                # 构造截取框
                crop_box = np.array([nx1, ny1, nx2, ny2])
                # 人脸框相对于截取图片的偏移量并做归一化处理
                offset_x1 = (gt_x1 - nx1) / float(size)
                offset_y1 = (gt_y1 - ny1) / float(size)
                offset_x2 = (gt_x2 - nx2) / float(size)
                offset_y2 = (gt_y2 - ny2) / float(size)
                cropped_img = img[ny1:ny2, nx1:nx2, :]
                resized_img = cv2.resize(cropped_img, (12, 12), interpolation=cv2.INTER_LINEAR)
                # gt_box扩充一个维度作为iou输入
                iou = cal_ious(crop_box, gt_box.reshape(1, -1))
                # pos样本
                if iou >= 0.65:
                    save_file = os.path.join(config.pnet_pos_dir, '%s.jpg' % id_positive)
                    f_pos.write(config.pnet_pos_dir + '/%s.jpg' % id_positive +
                                ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_img)
                    id_positive += 1
                # part样本
                elif iou >= 0.4:
                    save_file = os.path.join(config.pnet_part_dir, '%s.jpg' % id_part)
                    f_part.write(config.pnet_part_dir + '/%s.jpg' % id_part +
                                 ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_img)
                    id_part += 1

    print('%s 个图片已处理，正样本数量：%s  部分样本数量: %s  负样本数量: %s' % (img_done, id_positive, id_part, id_negative))
    f_pos.close()
    f_neg.close()
    f_part.close()


def gen_hard_examples(input_size):
    """通过PNet或RNet生成下一个网络的输入"""
    model_paths = [config.pnet_checkpoint_dir,
                   config.rnet_checkpoint_dir,
                   config.onet_checkpoint_dir]
    detectors = [None, None, None]
    detectors[0] = SubDetector(model_paths[0])

    if input_size == 12:
        output_size = 24
        net_data_dir = config.rnet_data_dir
    elif input_size == 24:
        output_size = 48
        net_data_dir = config.onet_data_dir
        detectors[1] = SubDetector(model_paths[1])
    else:
        print("Invalid image size")
        sys.exit(1)

    # 处理后的图片存放地址
    neg_dir = os.path.join(net_data_dir, 'negative')
    pos_dir = os.path.join(net_data_dir, 'positive')
    part_dir = os.path.join(net_data_dir, 'part')

    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 读取文件的image和box对应函数在utils中
    data = gen_data_from_wider()
    mtcnn_detector = MainDetector(detectors)

    save_file = os.path.join(net_data_dir, 'detections.pkl')

    if not os.path.exists(save_file):
        # 将data制作成迭代器
        print('开始识别')
        test_data = TestLoader(data['img_paths'])
        detecte_results, _ = mtcnn_detector.detect_imgs(test_data)
        print('完成识别')

        with open(save_file, 'wb') as f:
            pickle.dump(detecte_results, f, 1)
    else:
        print('已完成识别')

    print('开始生成图像')
    save_hard_examples(output_size, data, neg_dir, pos_dir, part_dir, net_data_dir)


def save_hard_examples(output_size, data, neg_dir, pos_dir, part_dir, net_data_dir):
    """将网络识别的box用来裁剪原图像作为下一个网络的输入"""

    img_paths = data['img_paths']
    all_gt_boxes = data['gt_boxes']
    num_images = len(img_paths)
    all_pred_boxes = pickle.load(open(os.path.join(net_data_dir, 'detections.pkl'), 'rb'))
    
    assert len(all_pred_boxes) == num_images, 'ERROR: len(all_pred_boxes) != num_images,'

    if output_size == 24:
        f_pos_img_list = open(config.rnet_pos_img_list, 'w')
        f_neg_img_list = open(config.rnet_neg_img_list, 'w')
        f_part_img_list = open(config.rnet_part_img_list, 'w')
    if output_size == 48:
        f_pos_img_list = open(config.onet_pos_img_list, 'w')
        f_neg_img_list = open(config.onet_neg_img_list, 'w')
        f_part_img_list = open(config.onet_part_img_list, 'w')

    neg_cnt, pos_cnt, part_cnt = 0, 0, 0

    for img_path, pred_boxes, gt_boxes in tqdm(zip(img_paths, all_pred_boxes, all_gt_boxes), total=len(img_paths)):
        gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 4)

        if pred_boxes.shape[0] == 0 or len(gt_boxes) == 0:
            continue
        img = cv2.imread(img_path)
        # 转换成正方形
        pred_boxes = convert_to_square(pred_boxes)
        pred_boxes[:, 0:4] = np.round(pred_boxes[:, 0:4])
        neg_cnt_for_current_img = 0
        for pred_box in pred_boxes:

            x_left, y_top, x_right, y_bottom, _ = pred_box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的和超出边界的预测框
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue
            
            ious = cal_ious(pred_box, gt_boxes)
            
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (output_size, output_size),
                                    interpolation=cv2.INTER_LINEAR)
            # 保存一定数量的负样本
            if np.max(ious) < 0.3 and neg_cnt_for_current_img < 60:
                save_file = os.path.join(neg_dir, "%s.jpg" % neg_cnt)
                f_neg_img_list.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                neg_cnt += 1
                neg_cnt_for_current_img += 1
            # 保存正样本和部分样本
            else:
                idx = np.argmax(ious)
                assigned_gt = gt_boxes[idx]
                x1, y1, x2, y2 = assigned_gt

                # 偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # positive sample
                if np.max(ious) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % pos_cnt)
                    f_pos_img_list.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    pos_cnt += 1

                # part sample
                elif np.max(ious) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % part_cnt)
                    f_part_img_list.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    part_cnt += 1

    f_neg_img_list.close()
    f_part_img_list.close()
    f_pos_img_list.close()

    print('图像生成完成')