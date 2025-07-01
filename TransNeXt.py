'''
这个版本解决一下一直以来训练文件臃肿不堪的情况
尽量优雅地重置网络，使得尽可能精简，但做不到，草
'''
import argparse
import logging
import os
import random
import sys
import time
from Config_for_datasets import Config
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from models.TransNeXt.model_final import TransN
from fvcore.nn import  parameter_count_table
from Config_for_datasets.Config import CONFIG_HSIC
#导入我自己制作的datasloader
from Mydataset.High_Dataset import MyDataset
from torch.utils.data import DataLoader
from function.val_function import validate
from function.train_function import train
from function.test_function import test
# from function.Draw_featuremaps import draw_featuremap


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser("TransN")
parser.add_argument('--exp_name', type=str, default='TransN', help='experiment name')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='checkpoints direction')#保存模型的地址
# Training Settings
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--iters', type=int, default=10, help='number of experiment repeats')
parser.add_argument('--learning_rate', type=float, default=0.005, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--seed', type=int, default=0, help='training seed')#xin我们没有设置种子，我觉得设置种子是投机取巧地方法，容易翻车，
parser.add_argument('--step_size', type=int, default=10, help='step_size')
parser.add_argument('--gamma', type=float, default=0.8, help='gamma')

# Dataset Settings
parser.add_argument('--data_root', type=str, default='./datasets/', help='datasets dir')
parser.add_argument('--classes', type=int, default=14, help='datasets classes')
parser.add_argument('--datasets', type=str, default='Loukia', help='path to the datasets')
parser.add_argument('--ablation', type=str, default='1', help='ablation')
parser.add_argument('--windows', type=int, default=11, help='size of patches')
parser.add_argument('--cutout', action='store_true', help='use cutout') #这两个参数暂时用不到，留作备用，先不删除
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')


args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
#固定随机因子，暂且留下，用时方便
# utils.set_seed(args.seed)

#需要一个函数来生成新的网络参数，同时，还需重置优化器和计划器
def get_model():
    model = TransN(CONFIG_HSIC[args.datasets]['bands'], CONFIG_HSIC[args.datasets]['num_classes'],
                   args.windows)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size,
    #                                 gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))
    return model, optimizer, scheduler

#训练选中的bestpath
def main():
    # Define Dataset
    # 准备训练集
    train_dataset = MyDataset(CONFIG_HSIC[args.datasets]['train_data_path'],
                              CONFIG_HSIC[args.datasets]['train_label_path'])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    # 准备验证集
    val_dataset = MyDataset(CONFIG_HSIC[args.datasets]['val_data_path'],
                            CONFIG_HSIC[args.datasets]['val_label_path'])
    # 加载验证集
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True, drop_last=True)
    # 准备验证集
    test_dataset = MyDataset(CONFIG_HSIC[args.datasets]['test_data_path'],
                            CONFIG_HSIC[args.datasets]['test_label_path'])
    # 加载验证集
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=True, drop_last=True)
    print(CONFIG_HSIC[args.datasets]['bands'])
    print(CONFIG_HSIC[args.datasets]['num_classes'])
    #创建模型
    # model, optimizer, scheduler = get_model()
    # model = model.to(args.device)
    # print(model)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # 保存最终的模型
    final_model = None
    final_OA = 0
    final_AA = 0
    final_kappa = 0
    final_report = 0
    eva_time = 0
    # 创建一个列表来存储各个迭代模型的OA
    OA_list = []
    AA_list = []
    Kappa_list = []
    OA_AA_Kappa_list = []
    Class_Acc = np.zeros((args.iters, CONFIG_HSIC[args.datasets]['num_classes']))

    # 接下来进入迭代的大循环
    for index_iter in range(args.iters):
        print('iter:', index_iter)
        # 每次迭代开始之前重置模型参数以及优化器等，这里用了个笨办法，重新申请了模型
        model, optimizer, scheduler = get_model()
        model = model.to(args.device)

        # 训练模型
        iter_model = train(args, index_iter, model, train_loader, val_loader, criterion, optimizer, scheduler,
                           args.epochs)
        # 测试模型
        iter_OA, iter_AA, iter_kappa, iter_report, acc_for_each_class, time = test(args, index_iter, test_loader,
                                                                                   iter_model)

        eva_time += time
        OA_list.append(iter_OA)
        AA_list.append(iter_AA)
        Kappa_list.append(iter_kappa)
        OA_AA_Kappa_list.append([iter_OA, iter_AA, iter_kappa])

        Class_Acc[index_iter, :] = acc_for_each_class

        # 保存最优迭代模型
        if final_OA < iter_OA:
            final_OA = iter_OA
            print("Best OA by now")
            final_model = iter_model  # 注意要保存当次迭代模型
            final_AA = iter_AA
            final_kappa = iter_kappa
            final_report = iter_report

    # 统计模型参数量
    print('======================================================')
    num_params = sum(param.numel() for param in final_model.parameters())
    print("模型参数量：")
    print(num_params)
    print("模型参数量核算：")
    print(parameter_count_table(final_model))
    print("模型infer速度：")
    print(eva_time / args.iters)

    # print("================Final 整体测试集上的报告=================")
    # print(OA_list)
    # print(final_report)
    # print("Final整体测试集上的OA:{}".format(final_OA))
    # print("Final整体测试集上的AA:{}".format(final_AA))
    # print("Final整体测试集上的kappa系数:{}".format(final_kappa))
    #
    # print("================均值方差=================")
    # OA_list = np.array(OA_list)
    # AA_list = np.array(AA_list)
    # Kappa_list = np.array(Kappa_list)
    #
    # combined_array = np.column_stack((OA_list, AA_list, Kappa_list))
    # file_path = 'txt/' + args.exp_name + '_' + args.datasets + '_results.txt'
    # if os.path.isfile(file_path) is False:
    #     open(file_path, 'wt')
    # np.savetxt(file_path, combined_array, fmt='%.4f', delimiter='\t')
    #
    # mean_OA = np.mean(OA_list, axis=0)
    # mean_AA = np.mean(AA_list, axis=0)
    # mean_Kappa = np.mean(Kappa_list, axis=0)
    # std_OA = np.std(OA_list, axis=0)
    # std_AA = np.std(AA_list, axis=0)
    # std_Kappa = np.std(Kappa_list, axis=0)
    # print("OA: Mean =", mean_OA, ", Std Dev =", std_OA)
    # print("AA: Mean =", mean_AA, ", Std Dev =", std_AA)
    # print("Kappa: Mean =", mean_Kappa, ", Std Dev =", std_Kappa)

    # 类别以及OA AA KAPPA 的平均值±标准差
    print("================Final 整体测试集上的报告=================")
    print(final_report)
    print(OA_list)
    # 计算每个类别的平均
    mean_accuracies = np.mean(Class_Acc, axis=0)
    # 计算每个类别的标准差
    std_deviations = np.std(Class_Acc, axis=0, ddof=1)  # 使用样本标准差
    # 以平均准确率±标准差的形式输出
    for i in range(CONFIG_HSIC[args.datasets]['num_classes']):
        print(f"类别{i + 1} : {mean_accuracies[i] * 100:.2f} ± {std_deviations[i] * 100:.2f}")

    OA_AA_Kappa_list = np.array(OA_AA_Kappa_list)
    # 计算每个类别的平均
    mean_accuracies = np.mean(OA_AA_Kappa_list, axis=0)
    # 计算每个类别的标准差
    std_deviations = np.std(OA_AA_Kappa_list, axis=0, ddof=1)  # 使用样本标准差
    # 以平均准确率±标准差的形式输出
    name_list = ['OA', 'AA', 'Kappa']
    for i in range(3):
        print(f"{name_list[i]} : {mean_accuracies[i] * 100:.2f} ± {std_deviations[i] * 100:.2f}")

    # 保存最终模型
    save_path = 'model_pth/' + args.datasets + '/' + args.exp_name + '_' + 'ab' + '{' + str(
        args.ablation) + '}' + args.datasets + '.pth'
    if os.path.isdir('model_pth/' + args.datasets) is False:
        os.makedirs('model_pth/' + args.datasets)
    if os.path.isfile(save_path) is False:
        open(save_path, 'wt')
    torch.save(final_model, save_path)
    print('模型已保存')

    # 把画图的函数也给加上，每次跑完把feature_map也给画出来并保存
    # draw_featuremap(args.device, args.datasets, final_model, args.windows, args.exp_name)


if __name__ == '__main__':
    main()

