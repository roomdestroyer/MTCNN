import os
import sys
from config import pnet_mixed_img_list, pnet_tfrecord_file
import numpy as np
from train.utils import augment_image, read_tfrecord
import tensorflow as tf
import config
import shutil


def train_model(net, checkpoint_dir, num_epoch, size, log_step, base_lr):
    """ 训练模型 """
    if size == 12:
        # 读取img_list文件
        f = open(pnet_mixed_img_list, 'r')
        num_samples = len(f.readlines())
        # 从tfrecord读取数据
        batch_sizes = [config.batch_size]
        tfrecord_files = [pnet_tfrecord_file]
        log_file = config.train_pnet_log_file
    elif size == 24 or size == 48:
        if size == 24:
            # 读取img_list文件
            f_pos = open(config.rnet_pos_img_list, 'r')
            f_neg = open(config.rnet_neg_img_list, 'r')
            f_part = open(config.rnet_part_img_list, 'r')
            f_landmark = open(config.rnet_landmark_img_list, 'r')
            num_samples = len(f_pos.readlines()) + len(f_neg.readlines()) + \
                          len(f_part.readlines()) + len(f_landmark.readlines())
            tfrecord_file_pos = config.rnet_tfrecord_file_pos
            tfrecord_file_neg = config.rnet_tfrecord_file_neg
            tfrecord_file_part = config.rnet_tfrecord_file_part
            tfrecord_file_landmark = config.rnet_tfrecord_file_landmark
            tfrecord_files = [tfrecord_file_pos, tfrecord_file_neg, tfrecord_file_part, tfrecord_file_landmark]
            log_file = config.train_rnet_log_file
        else:
            # 读取img_list文件
            f_pos = open(config.onet_pos_img_list, 'r')
            f_neg = open(config.onet_neg_img_list, 'r')
            f_part = open(config.onet_part_img_list, 'r')
            f_landmark = open(config.onet_landmark_img_list, 'r')
            num_samples = len(f_pos.readlines()) + len(f_neg.readlines()) + \
                          len(f_part.readlines()) + len(f_landmark.readlines())
            tfrecord_file_pos = config.onet_tfrecord_file_pos
            tfrecord_file_neg = config.onet_tfrecord_file_neg
            tfrecord_file_part = config.onet_tfrecord_file_part
            tfrecord_file_landmark = config.onet_tfrecord_file_landmark
            tfrecord_files = [tfrecord_file_pos, tfrecord_file_neg, tfrecord_file_part, tfrecord_file_landmark]
            log_file = config.train_onet_log_file
        # 调整各类数据的比例，确保每一个batch各种数据的占比相同
        pos_weight, neg_weight, part_weight, landmark_weight = 1.0 / 6, 3.0 / 6, 1.0 / 6, 1.0 / 6
        pos_batch_size = int(np.ceil(config.batch_size * pos_weight))
        neg_batch_size = int(np.ceil(config.batch_size * neg_weight))
        part_batch_size = int(np.ceil(config.batch_size * part_weight))
        landmark_batch_size = int(np.ceil(config.batch_size * landmark_weight))
        batch_sizes = [pos_batch_size, neg_batch_size, part_batch_size, landmark_batch_size]
    else:
        print("Invalid image size")
        sys.exit(1)

    # 从tfrecord中构建数据集
    dataset = read_tfrecord(tfrecord_files, batch_sizes, net.img_size)
    # 定义一个全局步数变量，用于记录优化器的迭代次数
    global_step = tf.Variable(0, trainable=False)
    # 计算学习率衰减的边界值(以batch号来表示)
    decay_batches = [int(decay_epoch * num_samples / config.batch_size) for decay_epoch in config.decay_epochs]
    # 计算学习率随着训练次数而变化的值，假设decay_epochs=[5,15,20]，那么学习率被三个衰减点分为四个区间
    lr_values = [base_lr * (0.1 ** x) for x in range(0, len(config.decay_epochs) + 1)]
    # 创建一个分段常数衰减的学习率策略，参数包括衰减边界和不同阶段的学习率值
    lr_schedule = tf.optimizers.schedules.PiecewiseConstantDecay(decay_batches, lr_values)(global_step)
    # 创建一个SGD优化器，momentum参数取值范围为[0, 1]，有助于在训练过程中减小优化器的振荡，从而实现更快的收敛
    sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    # 训练循环
    for epoch in range(1, num_epoch + 1):
        # 训练批次（batch）循环
        for step, (image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array) \
                in enumerate(dataset, start=1):
            image_batch_array = augment_image(image_batch_array)
            # 创建一个梯度计算的上下文，并且计算损失和准确度
            with tf.GradientTape() as tape:
                # 分类损失、边界框损失、关键点损失、L2正则化损失（防过拟合）、准确率
                classification_loss, bbox_loss, landmark_loss, batch_accuracy = \
                    net(image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array, training=True)
                # 计算总损失，总损失是不同损失项的加权和
                total_loss = config.classification_loss_weight * classification_loss + \
                             config.bbox_loss_weight * bbox_loss + \
                             config.landmark_loss_weight * landmark_loss
                # 计算总损失相对于模型可训练参数的梯度
                grads = tape.gradient(total_loss, net.trainable_variables)
                # 更新模型参数，这一步使用了SGD来更新网络权重
                sgd_optimizer.apply_gradients(zip(grads, net.trainable_variables))

            # 展示训练过程
            if step % log_step == 0:
                train_info = "epoch: %d/%d; step: %d\n" % (epoch, num_epoch, step) + \
                             "batch acc: %3f; cls loss: %4f; bbox loss: %4f; landmark loss: %4f; total loss: %4f\n" % \
                             (batch_accuracy, classification_loss, bbox_loss, landmark_loss, total_loss)
                # print(train_info)
                # 将训练记录写入到日志文件中
                with open(log_file, 'a') as f:
                    f.write(train_info)
                    f.close()

    # 如果检查点文件存在则将其删除
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    # 保存模型
    net.save(checkpoint_dir, save_format='tf')
