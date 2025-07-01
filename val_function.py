'''
在val验证当前model的函数
输入：model,valloader,
返回：OA
'''
import torch
from sklearn import metrics


def validate(model, loader):
    #将模型设置为评估模式
    model.eval()
    # 创建一个predictions来记录我们预测的结果
    predictions = torch.FloatTensor().cuda()
    # labels是原本的标签
    labels = torch.FloatTensor().cuda()
    # 测试部分主体
    with torch.no_grad():
        for (X, y) in loader:
            X, y = X.cuda(), y.cuda()
            outs = model(X)
            # loss = self.criterion(outs, y)
            # 计算准确率等参数
            batch_prediction = outs.argmax(1).cuda()
            predictions = torch.cat((predictions, batch_prediction), 0)
            labels = torch.cat((labels, y), 0)

    # metrics最后要转换成numpy，但是从cuda是没法直接转成numpy的，所以要转到CPU上面
    predictions = predictions.cpu()
    labels = labels.cpu()
    OA = metrics.accuracy_score(labels, predictions) * 100
    OA = round(OA, 2)  #用于数字的四舍五入
    return OA