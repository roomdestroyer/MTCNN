import cv2
import os
from detection.sub_detector import SubDetector
from detection.main_detector import MainDetector
import config
import shutil
import tqdm


def test_videos(test_mode='pro'):
    """识别视频中的人脸框并将结果输出为视频文件"""
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

    if os.path.exists(config.output_videos_dir):
        shutil.rmtree(config.output_videos_dir)
    os.mkdir(config.output_videos_dir)

    if not os.path.exists(config.input_videos_dir):
        print('Invalid videos input directory')
        exit(1)

    # 识别视频文件夹下的每一个视频
    for item in os.listdir(config.input_videos_dir):
        input_video_path = os.path.join(config.input_videos_dir, item)
        output_video_path = os.path.join(config.output_videos_dir, item)

        # 捕获视频文件
        cap = cv2.VideoCapture(input_video_path)
        # mp4c是一种常用的视频编码格式，用于实现高质量的视频压缩，fourcc表示视频编码格式的四字节代码(four-character code)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 从视频对象中读取下一帧，如果读取成功则ret返回True
        ret, frame = cap.read()
        height, width, _ = frame.shape
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
        
        # 获取视频的总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 设置帧间隔，每5帧进行一次人脸检测
        frame_counter = -1
        frame_interval = 2
        boxes_c, landmarks = None, None
        with tqdm.tqdm(total=total_frames, desc=f"Processing {item}") as pbar:
            while ret:
                frame_counter += 1
                pbar.update(1)
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_counter % frame_interval == 0:
                    boxes_c, landmarks = mtcnn_detector.detect_img(frame)
                frame = draw(boxes_c, landmarks, frame)
                out.write(frame)

            cap.release()
            out.release()


def draw(boxes_c, landmarks, frame):
    """画人脸框和关键点"""
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # 画人脸框
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]), 
                        (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        # 判别为人脸的置信度
        cv2.putText(frame, '{:.2f}'.format(score), 
                    (corpbbox[0], corpbbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, None, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    # 画关键点
    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i]) // 2):
                cv2.circle(frame, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))

    return frame
