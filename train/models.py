import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.regularizers import l2
from train.utils import cls_ohem, bbox_ohem, landmark_ohem, cal_accuracy


class PReLU_(Layer):
    def __init__(self, **kwargs):
        """self.alphas用于保存可学习的参数alpha"""
        super(PReLU_, self).__init__(**kwargs)
        self.alphas = None

    def build(self, input_shape):
        """alphas的形状与输入张量的最后一维相同，即每个通道都有一个对应的alpha值"""
        self.alphas = self.add_weight(name='alphas', shape=(input_shape[-1],),
                                      initializer=tf.constant_initializer(0.25), dtype=tf.float32)

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = tf.multiply(self.alphas, (inputs - tf.abs(inputs))) * 0.5
        return pos + neg


class PNet(Model):
    def __init__(self, training=True):
        super(PNet, self).__init__()
        self.net_name = 'pnet'
        self.img_size = 12
        self.training = training

        self.conv1 = Conv2D(10, 3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu1 = PReLU_(name='prelu1')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

        self.conv2 = Conv2D(16, 3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu2 = PReLU_(name='prelu2')

        self.conv3 = Conv2D(32, 3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu3 = PReLU_(name='prelu3')

        self.conv4_1 = Conv2D(2, 1, activation='softmax', padding='valid')
        self.conv4_2 = Conv2D(4, 1, padding='valid')
        self.conv4_3 = Conv2D(10, 1, padding='valid')

    def call(self, inputs, label=None, bbox_target=None, landmark_target=None, **kwargs):
        self.training = kwargs.pop('training', self.training)
        net = self.conv1(inputs)
        net = self.prelu1(net)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.prelu2(net)
        net = self.conv3(net)
        net = self.prelu3(net)
        
        conv4_1 = self.conv4_1(net)
        bbox_pred = self.conv4_2(net)
        landmark_pred = self.conv4_3(net)

        if self.training:
            cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')
            cls_loss = cls_ohem(cls_prob, label)

            bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)

            landmark_pred = tf.squeeze(landmark_pred, [1, 2], name='landmark_pred')
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)

            accuracy = cal_accuracy(cls_prob, label)
            return cls_loss, bbox_loss, landmark_loss, accuracy
        else:
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
            landmark_pred_test = tf.squeeze(landmark_pred, axis=0)
            return cls_pro_test, bbox_pred_test, landmark_pred_test


class RNet(Model):
    def __init__(self, training=True):
        super(RNet, self).__init__()
        self.net_name = 'rnet'
        self.img_size = 24
        self.training = training

        self.conv1 = Conv2D(28, 3, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu1 = PReLU_()
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        self.conv2 = Conv2D(48, 3, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu2 = PReLU_()
        self.pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')

        self.conv3 = Conv2D(64, 2, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu3 = PReLU_()

        self.flatten = Flatten()
        self.fc1 = Dense(128)
        self.prelu4 = PReLU_()

        self.cls_fc = Dense(2, activation=tf.nn.softmax)
        self.bbox_fc = Dense(4, activation=None)
        self.landmark_fc = Dense(10, activation=None)

    def call(self, inputs, label=None, bbox_target=None, landmark_target=None, **kwargs):
        self.training = kwargs.pop('training', self.training)
        net = self.conv1(inputs)
        net = self.prelu1(net)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.prelu2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.prelu3(net)
        fc_flatten = self.flatten(net)
        fc1 = self.fc1(fc_flatten)
        fc1 = self.prelu4(fc1)

        cls_prob = self.cls_fc(fc1)
        bbox_pred = self.bbox_fc(fc1)
        landmark_pred = self.landmark_fc(fc1)

        if self.training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            return cls_loss, bbox_loss, landmark_loss, accuracy
        else:
            return cls_prob, bbox_pred, landmark_pred


class ONet(Model):
    def __init__(self, training=True):
        super(ONet, self).__init__()
        self.net_name = 'onet'
        self.img_size = 48
        self.training = training

        self.conv1 = Conv2D(32, 3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu1 = PReLU_()
        self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')

        self.conv2 = Conv2D(64, 3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu2 = PReLU_()
        self.pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')

        self.conv3 = Conv2D(64, 3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu3 = PReLU_()
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

        self.conv4 = Conv2D(128, 2, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.0005), padding='valid')
        self.prelu4 = PReLU_()

        self.flatten = Flatten()
        self.fc1 = Dense(256)
        self.prelu5 = PReLU_()
        self.cls_fc = Dense(2, activation=tf.nn.softmax)
        self.bbox_fc = Dense(4, activation=None)
        self.landmark_fc = Dense(10, activation=None)

    def call(self, inputs, label=None, bbox_target=None, landmark_target=None, **kwargs):
        self.training = kwargs.pop('training', True)
        net = self.conv1(inputs)
        net = self.prelu1(net)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.prelu2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.prelu3(net)
        net = self.pool3(net)
        net = self.conv4(net)
        net = self.prelu4(net)
        fc_flatten = self.flatten(net)
        fc1 = self.fc1(fc_flatten)
        fc1 = self.prelu5(fc1)

        cls_prob = self.cls_fc(fc1)
        bbox_pred = self.bbox_fc(fc1)
        landmark_pred = self.landmark_fc(fc1)

        if self.training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            return cls_loss, bbox_loss, landmark_loss, accuracy
        else:
            return cls_prob, bbox_pred, landmark_pred
