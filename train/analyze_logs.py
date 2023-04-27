import os
import numpy as np
import matplotlib.pyplot as plt
import config


def moving_average(data, window_size=1):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')


def analyze_train_log(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    losses = []
    accs = []
    current_epoch_losses = []
    current_epoch_accs = []
    current_epoch = 1

    for line in lines:
        if line.startswith('epoch'):
            parts = line.split(';')
            epoch = int(parts[0].split(':')[1].strip().split('/')[0])
            if epoch != current_epoch:
                if current_epoch_losses and current_epoch_accs:
                    losses.append(np.mean(current_epoch_losses))
                    accs.append(np.mean(current_epoch_accs))
                    current_epoch_losses = []
                    current_epoch_accs = []
                current_epoch = epoch

        elif line.startswith('batch acc'):
            parts = line.split(';')
            acc = float(parts[0].split(':')[1].strip())
            current_epoch_accs.append(acc)
            total_loss = float(parts[-1].split(':')[1].strip())
            current_epoch_losses.append(total_loss)

    if current_epoch_losses and current_epoch_accs:
        losses.append(np.mean(current_epoch_losses))
        accs.append(np.mean(current_epoch_accs))

    return losses, accs


def plot_metrics(losses, accs, idx):

    window_size = 1
    losses = moving_average(losses, window_size=window_size)
    accs = moving_average(accs, window_size=window_size)

    epochs = range(1, len(losses) + 1)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if idx == 0:
        plt.title('Training Loss & Accuracy in PNet')
        fig.tight_layout()
        plt.savefig(config.train_pnet_metrics_file)
    elif idx == 1:
        plt.title('Training Loss & Accuracy in RNet')
        fig.tight_layout()
        plt.savefig(config.train_rnet_metrics_file)
    elif idx == 2:
        plt.title('Training Loss & Accuracy in ONet')
        fig.tight_layout()
        plt.savefig(config.train_onet_metrics_file)



def analyze_train_logs():
    for idx, log_file in enumerate([config.train_pnet_log_file, config.train_rnet_log_file, 
                                    config.train_onet_log_file]):
        if not os.path.exists(log_file):
            print(f"日志文件 '{log_file}' 不存在。")
        else:
            losses, accs = analyze_train_log(log_file)
            plot_metrics(losses, accs, idx)
