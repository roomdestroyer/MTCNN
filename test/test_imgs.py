import cv2
import os
from detection.sub_detector import SubDetector
from detection.main_detector import MainDetector
import config
import shutil


def test_imgs(test_mode='pro'):

    pnet_detector = SubDetector(config.checkpoint_dirs[0])
    rnet_detector, onet_detector = None, None

    if test_mode == 'p':
        pass
    elif test_mode == 'pr':
        rnet_detector = SubDetector(config.checkpoint_dirs[1])
    elif test_mode == 'pro':
        onet_detector = SubDetector(config.checkpoint_dirs[2])
    else:
        print('ERROR: Invalid test mode')
        exit(1)
    
    mtcnn_detector = MainDetector(detectors=[pnet_detector, rnet_detector, onet_detector])

    if os.path.exists(config.output_imgs_dir):
        shutil.rmtree(config.output_imgs_dir)
    os.mkdir(config.output_imgs_dir)
    if not os.path.exists(config.input_imgs_dir):
        os.mkdir(config.output_imgs_dir)

    # 选用图片
    for item in os.listdir(config.input_imgs_dir):
        img_path = os.path.join(config.input_imgs_dir, item)
        if not img_path.endswith(('.jpg', '.png')):
            continue
        img = cv2.imread(img_path)
        boxes_c, landmarks = mtcnn_detector.detect_img(img)
        
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # 画人脸框
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            # 判别为人脸的置信度
            cv2.putText(img, '{:.2f}'.format(score),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 画关键点
        if landmarks is not None:
            for i in range(landmarks.shape[0]):
                for j in range(len(landmarks[i]) // 2):
                    cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))

        cv2.imwrite(os.path.join(config.output_imgs_dir, item), img)
