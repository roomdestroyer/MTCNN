import os

"""preprocess"""

# 数据集根路径
current_working_directory = os.getcwd()
data_base_dir = os.path.join(current_working_directory, 'data')

# pnet, rnet, onet的数据目录
pnet_data_dir = os.path.join(data_base_dir, 'pnet_data')
rnet_data_dir = os.path.join(data_base_dir, 'rnet_data')
onet_data_dir = os.path.join(data_base_dir, 'onet_data')


# pnet的各种图片及其img_list文本文件的路径
pnet_pos_dir = os.path.join(pnet_data_dir, 'positive')
pnet_neg_dir = os.path.join(pnet_data_dir, 'negative')
pnet_part_dir = os.path.join(pnet_data_dir, 'part')
pnet_landmark_dir = os.path.join(pnet_data_dir, 'landmark')
pnet_pos_img_list = os.path.join(pnet_data_dir, 'positive_img_list.txt')
pnet_neg_img_list = os.path.join(pnet_data_dir, 'negative_img_list.txt')
pnet_part_img_list = os.path.join(pnet_data_dir, 'part_img_list.txt')
pnet_landmark_img_list = os.path.join(pnet_data_dir, 'landmark_img_list.txt')
pnet_mixed_img_list = os.path.join(pnet_data_dir, 'mixed_img_list.txt')
pnet_tfrecord_dir = os.path.join(pnet_data_dir, 'tfrecord')
pnet_tfrecord_file = os.path.join(pnet_tfrecord_dir, 'tfrecord')


# rnet的各种图片及其img_list文本文件的路径
rnet_pos_dir = os.path.join(rnet_data_dir, 'positive')
rnet_neg_dir = os.path.join(rnet_data_dir, 'negative')
rnet_part_dir = os.path.join(rnet_data_dir, 'part')
rnet_landmark_dir = os.path.join(rnet_data_dir, 'landmark')
rnet_pos_img_list = os.path.join(rnet_data_dir, 'positive_img_list.txt')
rnet_neg_img_list = os.path.join(rnet_data_dir, 'negative_img_list.txt')
rnet_part_img_list = os.path.join(rnet_data_dir, 'part_img_list.txt')
rnet_landmark_img_list = os.path.join(rnet_data_dir, 'landmark_img_list.txt')
rnet_tfrecord_dir = os.path.join(rnet_data_dir, 'tfrecord')
rnet_tfrecord_file_pos = os.path.join(rnet_tfrecord_dir, 'pos_img_list_tfrecord')
rnet_tfrecord_file_neg = os.path.join(rnet_tfrecord_dir, 'neg_img_list_tfrecord')
rnet_tfrecord_file_part = os.path.join(rnet_tfrecord_dir, 'part_img_list_tfrecord')
rnet_tfrecord_file_landmark = os.path.join(rnet_tfrecord_dir, 'landmark_img_list_tfrecord')


# onet的各种图片及其img_list文本文件的路径
onet_pos_dir = os.path.join(onet_data_dir, 'positive')
onet_neg_dir = os.path.join(onet_data_dir, 'negative')
onet_part_dir = os.path.join(onet_data_dir, 'part')
onet_landmark_dir = os.path.join(onet_data_dir, 'landmark')
onet_pos_img_list = os.path.join(onet_data_dir, 'positive_img_list.txt')
onet_neg_img_list = os.path.join(onet_data_dir, 'negative_img_list.txt')
onet_part_img_list = os.path.join(onet_data_dir, 'part_img_list.txt')
onet_landmark_img_list = os.path.join(onet_data_dir, 'landmark_img_list.txt')
onet_tfrecord_dir = os.path.join(onet_data_dir, 'tfrecord')
onet_tfrecord_file_pos = os.path.join(onet_tfrecord_dir, 'pos_img_list_tfrecord')
onet_tfrecord_file_neg = os.path.join(onet_tfrecord_dir, 'neg_img_list_tfrecord')
onet_tfrecord_file_part = os.path.join(onet_tfrecord_dir, 'part_img_list_tfrecord')
onet_tfrecord_file_landmark = os.path.join(onet_tfrecord_dir, 'landmark_img_list_tfrecord')

# WIDER数据集路径，WIDER图像的路径及其label
WIDER_base_dir = os.path.join(data_base_dir, 'WIDER')
WIDER_train_dir = os.path.join(WIDER_base_dir, 'WIDER_train')
WIDER_train_images_dir = os.path.join(WIDER_train_dir, 'images')

WIDER_labels_dir = os.path.join(WIDER_base_dir, 'wider_face_split')
wider_face_train_label = os.path.join(WIDER_labels_dir, 'wider_face_train.txt')
wider_face_train_bbx_gt = os.path.join(WIDER_labels_dir, 'wider_face_train_bbx_gt.txt')

# landmark数据集（lfw）的label
LFW_base_dir = os.path.join(data_base_dir, 'LFW')
lfw_train_labels = os.path.join(LFW_base_dir, 'trainImageList.txt')


""" train """

# 批次大小
batch_size = 384
# 三个网络的阈值
net_thresholds = [0.7, 0.8, 0.96]
# ohem策略中使用的有效样本比例
ohem_keep_ratio = 0.8
# 训练日志保存的位置
logs_dir = os.path.join(data_base_dir, 'logs')
train_pnet_log_file = os.path.join(logs_dir, 'train_pnet.log')
train_rnet_log_file = os.path.join(logs_dir, 'train_rnet.log')
train_onet_log_file = os.path.join(logs_dir, 'train_onet.log')
# 日志文件的解析输出
train_pnet_metrics_file = os.path.join(logs_dir, 'train_pnet_metrics')
train_rnet_metrics_file = os.path.join(logs_dir, 'train_rnet_metrics')
train_onet_metrics_file = os.path.join(logs_dir, 'train_onet_metrics')

# checkpoint保存位置
base_checkpoint_dir = 'model_checkpoints'
pnet_checkpoint_dir = os.path.join(data_base_dir, base_checkpoint_dir, 'pnet')
rnet_checkpoint_dir = os.path.join(data_base_dir, base_checkpoint_dir, 'rnet')
onet_checkpoint_dir = os.path.join(data_base_dir, base_checkpoint_dir, 'onet')
checkpoint_dirs = [pnet_checkpoint_dir, rnet_checkpoint_dir, onet_checkpoint_dir]

# 不同损失的权重
classification_loss_weight = 5
bbox_loss_weight = 2
landmark_loss_weight = 2

# 迭代次数
num_epochs = [60, 50, 40]
# 初始学习率
lr = 0.001

# 学习率减少的迭代次数
decay_epochs = [6, 16, 24]

# 经过多少epoch后显示log
log_step = 100


"""test"""

# 测试集所在路径
test_base_dir = os.path.join(data_base_dir, 'test')
# 测试图片放置位置
input_imgs_dir = os.path.join(test_base_dir, 'input_imgs')
# 测试图片输出位置
output_imgs_dir = os.path.join(test_base_dir, 'output_imgs')
# 测试视频放置位置
input_videos_dir = os.path.join(test_base_dir, 'input_videos')
# 测试视频输出位置
output_videos_dir = os.path.join(test_base_dir, 'output_videos')