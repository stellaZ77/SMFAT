'''
这里保存test代码，每次迭代调用一次测试代码
'''
import time
import sys
import torch
import numpy as np
from sklearn import metrics


def test(args, iter, test_loader, model):
    model.eval()
    # 创建一个predictions来记录我们预测的结果
    predictions = torch.FloatTensor().to(args.device)
    # labels是原本的标签
    labels = torch.FloatTensor().to(args.device)
    T1 = time.time()  #测试开始的时间
    with torch.no_grad():  # 是一个上下文管理器，被该语句 wrap 起来的部分将不会track梯度。
        for (patches, targets) in test_loader:
            patches = patches.to(args.device)
            targets = targets.to(args.device)
            output = model(patches)
            # 记录这一个batch的预测结果
            batch_prediction = output.argmax(1)
            predictions = torch.cat((predictions, batch_prediction), 0)
            labels = torch.cat((labels, targets), 0)
    T2 = time.time()  #测试结束的时间
    print('程序推理时间:%s秒' % ((T2 - T1)))
    tim = (T2 - T1)

    #开始计算各个结果参数
    predictions = predictions.cpu()
    labels = labels.cpu()
    print("---Iter:{} 整体测试集上的报告---".format(iter))
    report = metrics.classification_report(labels, predictions, digits=4)
    print(report)
    acc_for_each_class = metrics.precision_score(labels, predictions, average=None)
    # OA得单独拎出来，因为要计算最优模型
    test_OA = metrics.accuracy_score(labels, predictions)
    print("整体测试集上的OA:{}".format(test_OA))
    test_AA = np.mean(acc_for_each_class)
    print("整体测试集上的AA:{}".format(test_AA))
    test_kappa = metrics.cohen_kappa_score(labels, predictions)
    print("整体测试集上的kappa系数:{}".format(test_kappa))
    print("test_OA={}".format(round(test_OA, 4)))

    return test_OA, test_AA, test_kappa, report, acc_for_each_class, tim
    # return test_OA, test_AA, test_kappa, report, tim


# import time
# import torch
# import numpy as np
# from sklearn import metrics
#
#
# def test(args, iter, test_loader, model):
#     model.eval()
#     # 初始化 predictions 和 labels
#     predictions = torch.IntTensor().to(args.device)
#     labels = torch.IntTensor().to(args.device)
#
#     T1 = time.time()
#     with torch.no_grad():
#         for (patches, targets) in test_loader:
#             patches = patches.to(args.device)
#             targets = targets.to(args.device)
#             output = model(patches)
#             batch_prediction = output.argmax(1)
#             predictions = torch.cat((predictions, batch_prediction), 0)
#             labels = torch.cat((labels, targets), 0)
#     T2 = time.time()
#
#     # 计算推理时间
#     tim = T2 - T1
#     print(f"程序推理时间: {tim:.2f} 秒")
#
#     # 计算性能指标
#     predictions = predictions.cpu()
#     labels = labels.cpu()
#
#     print(f"---Iter:{iter} 整体测试集上的报告---")
#     report = metrics.classification_report(labels, predictions, digits=4)
#     print(report)
#
#     acc_for_each_class = metrics.precision_score(labels, predictions, average=None)
#     test_OA = metrics.accuracy_score(labels, predictions)
#     test_AA = np.mean(acc_for_each_class)
#     test_kappa = metrics.cohen_kappa_score(labels, predictions)
#
#     print(f"整体测试集上的OA: {test_OA:.4f}")
#     print(f"整体测试集上的AA: {test_AA:.4f}")
#     print(f"整体测试集上的kappa系数: {test_kappa:.4f}")
#
#     # 返回测试指标
#     return test_OA, test_AA, test_kappa, report, acc_for_each_class, tim