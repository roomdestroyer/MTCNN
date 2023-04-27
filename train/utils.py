import tensorflow as tf
import config


def cls_ohem(cls_prob, label):
    """计算分类损失"""
    zeros = tf.zeros_like(label)
    # num_cls_prob == batch_size * 2，表示一个batch中所有样本的二分类概率值
    num_cls_prob = tf.size(cls_prob)
    # 将无效样本（label-2）和部分样本（label为-1）的label设为0，表示"无人脸"
    label_gt = tf.where(tf.less(label, 0), zeros, label)
    label_gt = tf.cast(label_gt, tf.int32)

    # 将cls_prob重塑为一个二维矩阵，每一行包含两个元素：非人脸概率和人脸概率
    cls_prob_reshaped = tf.reshape(cls_prob, [num_cls_prob, -1])
    # 获取batch_size
    batch_size = tf.cast(tf.shape(cls_prob)[0], tf.int32)
    # 对于当前batch，偶数行是非人脸概率，奇数行是人脸概率
    neg_row_id = tf.range(batch_size) * 2
    # 如果一个样本实际标签是0，那么其实际概率就是对应的偶数行；如果一个样本实际标签是1，那么其实际概率就是对应的偶数行+1
    indices_ = neg_row_id + label_gt
    # 各个样本的真实标签对应的概率
    label_prob = tf.squeeze(tf.gather(cls_prob_reshaped, indices_))

    # 统计neg和pos的数量
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    valid_indices = tf.where(label < zeros, zeros, ones)
    num_valid = tf.reduce_sum(valid_indices)

    # 选取一定比例的损失值较高的样本（难例），这是OHEM的核心策略，可以使样本具有更好的泛化性和防止过拟合
    keep_num = tf.cast(num_valid * config.ohem_keep_ratio, dtype=tf.int32)
    # 计算损失值
    loss = -tf.math.log(label_prob + 1e-10) * valid_indices
    loss, _ = tf.math.top_k(loss, k=keep_num)

    return tf.reduce_mean(loss)


def bbox_ohem(bbox_pred, bbox_target, label):
    """计算box的损失"""
    # 计算有效样本的索引
    zeros = tf.zeros_like(label, dtype=tf.float32)
    ones = tf.ones_like(label, dtype=tf.float32)
    valid_indices = tf.where(tf.equal(tf.abs(label), 1), ones, zeros)
    # 计算保留的数据的个数
    num_valid = tf.reduce_sum(valid_indices)
    keep_num = tf.cast(num_valid * config.ohem_keep_ratio, dtype=tf.int32)
    # 计算使用ohem策略后的损失
    square_error = tf.square(bbox_pred - bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    square_error = square_error * valid_indices
    square_error, _ = tf.nn.top_k(square_error, k=keep_num)
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_target, label):
    """计算关键点损失"""
    # 计算有效样本的索引
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    valid_indices = tf.where(tf.equal(label, -2), ones, zeros)
    # 计算保留的数据的个数
    num_valid = tf.reduce_sum(valid_indices)
    keep_num = tf.cast(num_valid * config.ohem_keep_ratio, dtype=tf.int32)
    # 计算使用ohem策略后的损失
    square_error = tf.square(landmark_pred - landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    square_error = square_error * valid_indices
    square_error, _ = tf.nn.top_k(square_error, k=keep_num)
    return tf.reduce_mean(square_error)


def cal_accuracy(cls_probabilities, labels):
    """计算分类精度"""
    # 预测概率最高的类别，0代表“非人脸”，1代表“人脸”
    predictions = tf.argmax(cls_probabilities, axis=1)
    labels_int = tf.cast(labels, tf.int64)

    # 保留值大于等于0的标签，即正样本和负样本
    valid_labels_indices = tf.where(tf.greater_equal(labels_int, 0))
    picked_indices = tf.squeeze(valid_labels_indices)

    # 获得正样本和负样本
    valid_labels = tf.gather(labels_int, picked_indices)
    valid_predictions = tf.gather(predictions, picked_indices)

    # 计算分类精度
    accuracy = tf.reduce_mean(tf.cast(tf.equal(valid_labels, valid_predictions), tf.float32))
    return accuracy


def augment_image(image):
    """用于在训练时增强图像，提高模型的泛化性能"""
    # 对比度在原始对比度的 0.5 倍到 1.5 倍之间随机变化
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # 亮度在原始亮度的 -0.2 到 +0.2 之间随机变化
    image = tf.image.random_brightness(image, max_delta=0.2)
    # 色相在原始色相的 -0.2 到 +0.2 之间随机变化
    image = tf.image.random_hue(image, max_delta=0.2)
    # 和度在原始饱和度的 0.5 倍到 1.5 倍之间随机变化
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image


def read_tfrecord(tfrecord_files, batch_sizes, image_size):
    """读取tfrecord文件并创建数据集"""
    def parse_example(example):
        """用于解析tfrecord文件中的每个样本"""
        feature_spec = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/label': tf.io.FixedLenFeature([], tf.int64),
            'image/roi': tf.io.FixedLenFeature([4], tf.float32),
            'image/landmark': tf.io.FixedLenFeature([10], tf.float32)
        }
        parsed_features = tf.io.parse_single_example(example, feature_spec)
        image = tf.io.decode_raw(parsed_features['image/encoded'], tf.uint8)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = (tf.cast(image, tf.float32) - 127.5) / 128
        label = tf.cast(parsed_features['image/label'], tf.float32)
        roi = tf.cast(parsed_features['image/roi'], tf.float32)
        landmark = tf.cast(parsed_features['image/landmark'], tf.float32)
        return image, label, roi, landmark

    def create_tfrecord_dataset(file):
        """用于创建tfrecord文件的数据集"""
        dataset = tf.data.TFRecordDataset(file).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=10000)  # 增加这一行以随机打乱数据集中的样本
        return dataset

    if image_size == 12:
        # 创建一个随机打乱的数据集
        dataset = tf.data.Dataset.list_files(tfrecord_files, shuffle=True)
        # interleave函数可以在多个数据集之间交替取样，产生一个新的数据集
        dataset = dataset.interleave(
            create_tfrecord_dataset,
            cycle_length=len(tfrecord_files),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        # 对数据进行预处理和批处理，batch函数将数据集划分为batch_size的批次，prefetch允许模型在处理当前批次的数据时预取下一个批次的数据
        dataset = dataset.batch(config.batch_size)
    else:
        # 创建每个类别的数据集
        pos_dataset = create_tfrecord_dataset(tfrecord_files[0])
        neg_dataset = create_tfrecord_dataset(tfrecord_files[1])
        part_dataset = create_tfrecord_dataset(tfrecord_files[2])
        landmark_dataset = create_tfrecord_dataset(tfrecord_files[3])

        # 为每个类别的数据集应用批处理大小
        pos_dataset = pos_dataset.batch(batch_sizes[0], drop_remainder=True)
        neg_dataset = neg_dataset.batch(batch_sizes[1], drop_remainder=True)
        part_dataset = part_dataset.batch(batch_sizes[2], drop_remainder=True)
        landmark_dataset = landmark_dataset.batch(batch_sizes[3], drop_remainder=True)

        # 合并批次数据集
        dataset = tf.data.Dataset.zip((pos_dataset, neg_dataset, part_dataset, landmark_dataset))

        # 将不同类别的样本合并到同一个批次中，并将多个批次连接成一个数据集
        dataset = dataset.map(lambda pos, neg, part, landmark:
                              (tf.concat([pos[0], neg[0], part[0], landmark[0]], axis=0),
                               tf.concat([pos[1], neg[1], part[1], landmark[1]], axis=0),
                               tf.concat([pos[2], neg[2], part[2], landmark[2]], axis=0),
                               tf.concat([pos[3], neg[3], part[3], landmark[3]], axis=0)),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # prefetch允许模型在处理当前批次的数据时预取下一个批次的数据
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
