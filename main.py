from preprocess.utils import create_data_dir
from preprocess.gen_data import gen_pnet_data, gen_rnet_data, gen_onet_data
from train.train import train_pnet, train_rnet, train_onet
from train.analyze_logs import analyze_train_logs
from test.test_imgs import test_imgs
from test.test_videos import test_videos
import sys


if __name__ == '__main__':
    help = '用法: python main.py [ -create | -gen p | -gen r | -gen o | ' + \
           '-train p | -train r | -train o | -train logs | -test imgs | -test videos | -all]'
    if len(sys.argv) < 2:
        print(help)
        exit(1)
    
    if sys.argv[1] == '-create':
        # 在项目下载到本地后先创建会使用到的文件夹，目录树在config.py中定义
        create_data_dir()
    elif sys.argv[1] == '-gen':
        if len(sys.argv) < 3:
            print(help)
        elif sys.argv[2] == 'p':
            # 生成PNet的训练数据，目录树在config.py中定义
            gen_pnet_data()
        elif sys.argv[2] == 'r':
            # 生成RNet的训练数据，目录树在config.py中定义
            gen_rnet_data()
        elif sys.argv[2] == 'o':
            # 生成ONet的训练数据，目录树在config.py中定义
            gen_onet_data()
        else:
            print(help)
    elif sys.argv[1] == '-train':
        if len(sys.argv) < 3:
            print(help)
        elif sys.argv[2] == 'p':
            # 训练PNet
            train_pnet()
        elif sys.argv[2] == 'r':
            # 训练RNet
            train_rnet()
        elif sys.argv[2] == 'o':
            # 训练ONet
            train_onet()
        elif sys.argv[2] == 'logs':
            # 分析训练日志文件，相关路径在config.py中定义
            analyze_train_logs()
        else:
            print(help)
    elif sys.argv[1] == '-test':
        if len(sys.argv) < 3:
            print(help)
        elif sys.argv[2] == 'imgs':
            # 使用训练好的模型预测图片，图片输入路径和输出路径在config.py中定义
            test_imgs()
        elif sys.argv[2] == 'videos':
            test_videos()
        else:
            print(help)
    elif sys.argv[1] == '-all':
        # 完整执行一遍所有流程
        create_data_dir()
        # (M2 Pro)生成PNet数据的时间为：0小时45分钟45.03秒
        gen_pnet_data()
        # PNet的训练时间为：0小时42分钟51.55秒
        train_pnet()
        # 生成RNet数据的时间为：1小时42分钟30.17秒
        gen_rnet_data()
        # RNet的训练时间为：0小时46分钟50.61秒
        train_rnet()
        gen_onet_data()
        train_onet()
        analyze_train_logs()
        # 图片和视频的输入目录下必须要有文件
        test_imgs()
        test_videos()
    else:
        print(help)
