import os
import random
import sys
import tensorflow as tf
import cv2
from tqdm import tqdm
import config


def gen_tfrecords(img_size):
    """生成tfrecords文件"""
    # pnet只生成一个混合的tfrecords，rnet和onet要分别生成4个
    if img_size == 12:
        if not os.path.exists(config.pnet_tfrecord_dir):
            os.mkdir(config.pnet_tfrecord_dir)
        tf_filenames = [config.pnet_tfrecord_file]
        img_lists = [config.pnet_mixed_img_list]
    elif img_size == 24:
        if not os.path.exists(config.rnet_tfrecord_dir):
            os.mkdir(config.rnet_tfrecord_dir)
        tf_filenames = [config.rnet_tfrecord_file_pos, config.rnet_tfrecord_file_neg,
                        config.rnet_tfrecord_file_part, config.rnet_tfrecord_file_landmark]
        img_lists = [config.rnet_pos_img_list, config.rnet_neg_img_list,
                     config.rnet_part_img_list, config.rnet_landmark_img_list]
    elif img_size == 48:
        if not os.path.exists(config.onet_tfrecord_dir):
            os.mkdir(config.onet_tfrecord_dir)
        tf_filenames = [config.onet_tfrecord_file_pos, config.onet_tfrecord_file_neg,
                        config.onet_tfrecord_file_part, config.onet_tfrecord_file_landmark]
        img_lists = [config.onet_pos_img_list, config.onet_neg_img_list,
                     config.onet_part_img_list, config.onet_landmark_img_list]
    else:
        print("Invalid image size")
        sys.exit(1)

    # 一个img_list对应一个tfrecord文件
    for tf_filename, img_list in zip(tf_filenames, img_lists):
        # 从img_list中构建tensorflow所需要的examples
        img_examples = gen_img_examples(img_list)
        # 随机打乱数据集
        random.shuffle(img_examples)
        # 将构建得到的数据集写入到tfrecord文件中
        with tf.io.TFRecordWriter(tf_filename) as tfrecord_writer:
            # 将每一个img_example转换为tf_example，再写入到tfrecord文件中去
            for img_example in tqdm(img_examples):
                img_path = img_example['img_path']
                # 读取指定文件名的图像，并将其存储为一个numpy数组，然后将其转换为一个字节流
                image = cv2.imread(img_path)
                img_bytes = image.tobytes()
                # 将该image_bytes连同data_example转换成TensorFlow训练样本tf.train.Example
                tf_example = gen_tf_example(img_example, img_bytes)
                # 将给定的tf.train.Example对象example序列化为二进制字符串，然后通过TFRecordWriter对象tfrecord_writer将其写入到TFRecord文件中
                # 通过Example对象的SerializeToString()方法将其序列化为二进制字符串，然后通过TFRecordWriter对象的write()方法将其写入到TFRecord文件中
                tfrecord_writer.write(tf_example.SerializeToString())


def gen_img_examples(img_list):
    """ 从img_list中构建数TensorFlow所需的examples """
    f = open(img_list, 'r')
    dataset = []
    for line in tqdm(f.readlines()):
        line = line.strip().split(' ')
        # neg的box默认为0,part,pos的box只包含人脸框，landmark的box只包含关键点
        bbox = dict()
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0
        # 人脸框
        if len(line) == 6:
            bbox['xmin'] = float(line[2])
            bbox['ymin'] = float(line[3])
            bbox['xmax'] = float(line[4])
            bbox['ymax'] = float(line[5])
        # 关键点
        if len(line) == 12:
            bbox['xlefteye'] = float(line[2])
            bbox['ylefteye'] = float(line[3])
            bbox['xrighteye'] = float(line[4])
            bbox['yrighteye'] = float(line[5])
            bbox['xnose'] = float(line[6])
            bbox['ynose'] = float(line[7])
            bbox['xleftmouth'] = float(line[8])
            bbox['yleftmouth'] = float(line[9])
            bbox['xrightmouth'] = float(line[10])
            bbox['yrightmouth'] = float(line[11])

        data_example = dict()
        data_example['img_path'] = line[0]
        data_example['label'] = int(line[1])
        data_example['bbox'] = bbox

        dataset.append(data_example)

    return dataset


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def gen_tf_example(img_example, img_bytes):
    """ 转换成tfrecord接受形式 """
    class_label = img_example['label']
    bbox = img_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]

    # tf.train.Example表示一个训练样本，该样本包含多个特征，如标签、关键点、边界框等
    # tf.train.Features使用字典来表示该训练样本的特征值
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(img_bytes),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        'image/landmark': _float_feature(landmark)
    }))

    return example
