import tensorflow as tf
import numpy as np
import config
from train.models import PNet, RNet, ONet, PReLU_


class SubDetector:
    """识别多组图片"""
    def __init__(self, model_path):
        # PNet
        if model_path == config.checkpoint_dirs[0]:
            self.model_name = 'pnet'
            self.model = tf.keras.models.load_model(model_path, custom_objects={'PNet': PNet, 'PReLU_': PReLU_})
        # RNet
        elif model_path == config.checkpoint_dirs[1]:
            self.model_name = 'rnet'
            self.model = tf.keras.models.load_model(model_path, custom_objects={'RNet': RNet, 'PReLU_': PReLU_})
        # ONet
        elif model_path == config.checkpoint_dirs[2]:
            self.model_name = 'onet'
            self.model = tf.keras.models.load_model(model_path, custom_objects={'ONet': ONet, 'PReLU_': PReLU_})

    def predict(self, databatch, label=False, bbox_target=False, landmark_target=False, training=False):
        """将databatch分为多个小批次，分别对每个小批次进行预测"""
        if self.model_name == 'pnet':
            height, width, _ = databatch.shape
            databatch = tf.expand_dims(databatch, axis=0).numpy()
            cls_prob, bbox_pred, _ = self.model(databatch, label=None, bbox_target=None,
                                                landmark_target=None, training=False)
            return cls_prob.numpy(), bbox_pred.numpy()
        elif self.model_name == 'rnet' or self.model_name == 'onet':
            # 多个小批次被储存在minibatch列表中
            n = databatch.shape[0]
            batches = [databatch[i:i + config.batch_size] for i in range(0, n, config.batch_size)]

            cls_prob_list = []
            bbox_pred_list = []
            landmark_pred_list = []
            for data in batches:
                m = data.shape[0]
                real_size = config.batch_size

                if m < config.batch_size:
                    keep_inds = np.arange(m)
                    gap = config.batch_size - m
                    while gap >= len(keep_inds):
                        gap -= len(keep_inds)
                        keep_inds = np.concatenate((keep_inds, keep_inds))
                    if gap != 0:
                        keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                    data = data[keep_inds]
                    real_size = m

                cls_prob, bbox_pred, landmark_pred = self.model(data, label, bbox_target, landmark_target,
                                                                training=training)
                cls_prob_list.append(cls_prob[:real_size])
                bbox_pred_list.append(bbox_pred[:real_size])
                landmark_pred_list.append(landmark_pred[:real_size])

            return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(
                landmark_pred_list, axis=0)
