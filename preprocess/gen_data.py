import sys
import time
from preprocess.gen_ppn_data import gen_source_ppn_samples, gen_hard_examples
from preprocess.gen_landmark_data import gen_landmark_data
from preprocess.gen_tfrecords import gen_tfrecords
from preprocess.utils import pnet_mix_img_lists


def gen_pnet_data():
    start_time = time.time()
    print("\033[32m====================================================\033[0m")
    print("\033[32m开始生成PNet数据......\033[0m")
    print("\033[32m开始生成positive、part、negative样本......\033[0m")
    gen_source_ppn_samples()
    print("\033[32mpositive、part、negative样本生成完成，开始生成landmark、样本......\033[0m")
    gen_landmark_data(img_size=12)
    print("\033[32mlandmark样本生成完成，开始混合各种样本的labels......\033[0m")
    pnet_mix_img_lists()
    print("\033[32m混合完成，pnet的输入数据准备完成，开始生成tfrecord文件......\033[0m")
    gen_tfrecords(img_size=12)
    print("\033[32mtfrecord文件生成完成，PNet数据准备结束\033[0m")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"生成PNet数据的时间为：{int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒\n")


def gen_rnet_data():
    start_time = time.time()
    print("\033[32m====================================================\033[0m")
    print("\033[32m开始生成RNet数据......\033[0m")
    print("\033[32m开始生成hard examples......\033[0m")
    gen_hard_examples(input_size=12)
    print("\033[32mhard examples生成完成，开始生成landmark样本......\033[0m")
    gen_landmark_data(img_size=24)
    print("\033[32mlandmark 样本生成完成，开始生成tfrecords文件......\033[0m")
    gen_tfrecords(img_size=24)
    print("\033[32mtfrecord文件生成完成，RNet数据准备结束\033[0m")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"生成RNet数据的时间为：{int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒\n")


def gen_onet_data():
    start_time = time.time()
    print("\033[32m====================================================\033[0m")
    print("\033[32m开始生成ONet数据......\033[0m")
    print("\033[32m开始生成hard examples......\033[0m")
    gen_hard_examples(input_size=24)
    print("\033[32mhard examples生成完成，开始生成landmark样本......\033[0m")
    gen_landmark_data(img_size=48)
    print("\033[32mlandmark样本生成完成，开始生成tfrecords文件......\033[0m")
    gen_tfrecords(img_size=48)
    print("\033[32mtfrecord文件生成完成，ONet数据准备结束\033[0m")
    print("\033[32m====================================================\033[0m")
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"生成ONet数据的时间为：{int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒\n")
