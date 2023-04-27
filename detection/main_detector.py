from preprocess.utils import *
from tqdm import tqdm
import tensorflow as tf
from detection.utils import filter_boxes, resize_and_normalize_image, pad, calibrate_box, generate_bbox_from_pnet
import config


class MainDetector:
    """来生成人脸的图像"""

    def __init__(self, detectors):
        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]

    def detect_img(self, img):
        """用于测检测个图像"""
        boxes, boxes_c, landmark = None, None, None

        # pnet
        if self.pnet_detector:
            boxes, boxes_c, _ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]), np.array([])
        # rnet
        if self.rnet_detector:
            boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])
        # onet
        if self.onet_detector:
            boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

        return boxes_c, landmark

    def detect_imgs(self, test_data):
        """用于检测图像中的人脸，并返回检测到的人脸框和关键点信息"""
        # 用于储存检测到的人脸框和人脸关键点信息
        all_boxes, landmarks = [], []
        batch_id = 0
        num_of_img = test_data.size
        empty_array = np.array([])

        for databatch in tqdm(test_data, total=num_of_img):
            batch_id += 1
            im = databatch
            if self.pnet_detector:
                boxes, boxes_c, landmark = self.detect_pnet(im)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            if self.rnet_detector:
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            if self.onet_detector:
                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue

            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)

        return all_boxes, landmarks

    def detect_pnet(self, img):
        """在图像中检测人脸，并返回检测到的人脸位置和置信度"""
        resize_scale = 0.75
        # 将图像按照指定的比率缩放，并将其归一化
        img_resized = resize_and_normalize_image(img, resize_scale)
        img_height, img_width, _ = img_resized.shape
        # 存储所有检测到的人脸框
        all_boxes = []

        # 图像金字塔
        while min(img_height, img_width) > 12:
            # 针对缩放后的图像进行人脸检测，获取预测结果
            cls_map, reg_map = self.pnet_detector.predict(img_resized)
            boxes = generate_bbox_from_pnet(cls_map[:, :, 1], reg_map, resize_scale, config.net_thresholds[0])
            # 将图像按照指定的比率缩放，并将其归一化
            resize_scale *= 0.75
            img_resized = resize_and_normalize_image(img, resize_scale)
            img_height, img_width, _ = img_resized.shape

            if boxes.size == 0:
                continue

            # 执行非极大值抑制操作，留下重复低的边界框
            selected_indices = filter_boxes(boxes[:, :5], 0.5)
            boxes = boxes[selected_indices]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)
        # 将金字塔之后的box也进行非极大值抑制
        selected_indices = filter_boxes(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[selected_indices]
        boxes = all_boxes[:, :5]
        # box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None

    def detect_rnet(self, im, dets):
        img_size = 24
        h, w, c = im.shape
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        # 调整超出图像的box
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        delete_size = np.ones_like(tmpw) * 20
        ones = np.ones_like(tmpw)
        zeros = np.zeros_like(tmpw)
        num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
        cropped_ims = np.zeros((num_boxes, img_size, img_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
            if tmph[i] < 20 or tmpw[i] < 20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (img_size, img_size)) - 127.5) / 128
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > config.net_thresholds[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None, None

        keep = filter_boxes(boxes, 0.6)
        boxes = boxes[keep]
        # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
        boxes_c = calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None

    def detect_onet(self, im, dets):
        """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
        img_size = 48
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, img_size, img_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (img_size, img_size)) - 127.5) / 128

        batch_size = len(cropped_ims)
        label = tf.zeros(shape=(batch_size,), dtype=tf.float32)
        bbox_target = tf.zeros(shape=(batch_size, 4), dtype=tf.float32)
        landmark_target = tf.zeros(shape=(batch_size, 10), dtype=tf.float32)
        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims, label, bbox_target, landmark_target,
                                                               training=False)

        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > config.net_thresholds[2])[0]
        if len(keep_inds) > 0:

            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = calibrate_box(boxes, reg)

        boxes = boxes[filter_boxes(boxes, 0.1)]
        keep = filter_boxes(boxes_c, 0.1)
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark
