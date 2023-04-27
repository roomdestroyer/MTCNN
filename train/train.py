from train.models import PNet, RNet, ONet
from train.train_model import train_model
import config
import sys
import time


def train_pnet():
    start_time = time.time()
    net = PNet(training=True)
    num_epoch = config.num_epochs[0]
    checkpoint_dir = config.pnet_checkpoint_dir
    print("\033[32m====================================================\033[0m")
    print("\033[32m开始训练PNet\033[0m")
    train_model(net, checkpoint_dir, num_epoch, net.img_size, config.log_step, config.lr)
    print("\033[32mPNet训练完成\033[0m")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"PNet的训练时间为：{int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒\n")


def train_rnet():
    start_time = time.time()
    net = RNet(training=True)
    num_epochs = config.num_epochs[1]
    checkpoint_dir = config.rnet_checkpoint_dir
    print("\033[32m====================================================\033[0m")
    print("\033[32m开始训练RNet\033[0m")
    train_model(net, checkpoint_dir, num_epochs, net.img_size, config.log_step, config.lr)
    print("\033[32mRNet训练完成\033[0m")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"程序代码的运行时间为：{int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒\n")


def train_onet():
    start_time = time.time()
    net = ONet(training=True)
    num_epochs = config.num_epochs[2]
    checkpoint_dir = config.onet_checkpoint_dir
    print("\033[32m====================================================\033[0m")
    print("\033[32m开始训练ONet\033[0m")
    train_model(net, checkpoint_dir, num_epochs, net.img_size, config.log_step, config.lr)
    print("\033[32mONet训练完成\033[0m")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"程序代码的运行时间为：{int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒\n")


if __name__ == '__main__':
    if sys.argv[1] == 'p':
        train_pnet()
    if sys.argv[1] == 'r':
        train_rnet()
    if sys.argv[1] == 'o':
        train_onet()
