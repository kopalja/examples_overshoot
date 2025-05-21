import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt







def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='lightning_logs')
    args = parser.parse_args()


    data = {}
    for name in os.listdir(args.root):
        train_stats = pd.read_csv(os.path.join(args.root, name, 'version_1', 'training_stats.csv'))
        test_stats = pd.read_csv(os.path.join(args.root, name, 'version_1', 'validation_stats.csv'))

        train_key = 'train_shifted_to_base_loss' if 'train_shifted_to_base_loss' in train_stats.columns else 'train_basic_loss'
        data[name] = {
            "train_loss": train_stats[train_key],
            "test_acc1": test_stats["validation_acc1"],
            "test_acc5": test_stats["validation_acc5"],
        }
        
    for name, metrics in data.items():
        if name.startswith('baseline'):
            name = 'sgd'
        plt.plot(moving_average(metrics['train_loss'][1000:], window_size=30), label=name)

    plt.title('Resnet-50 on ImageNet') 
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('tmp.png')
    plt.clf()
    
    for name, metrics in data.items():
        if name.startswith('baseline'):
            name = 'sgd'
        plt.plot(metrics['test_acc1'], label=name)
    plt.title('Resnet-50 on ImageNet') 
    plt.xlabel('Epochs')
    plt.ylabel('top-1 accuracy')
    plt.legend()
    plt.savefig('tmp2.png')
    plt.clf()
    
    
    for name, metrics in data.items():
        if name.startswith('baseline'):
            name = 'sgd'
        plt.plot(metrics['test_acc5'], label=name)
    plt.title('Resnet-50 on ImageNet') 
    plt.xlabel('Epochs')
    plt.ylabel('top-5 accuracy')
    plt.legend()
    plt.savefig('tmp3.png')
