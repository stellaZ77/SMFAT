#这里是一个epoch发生的训练
import time
import logging
from .val_function import validate
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed time: %dh %dmin %ds' % (hour, minute, second))

# def train_epoch(args, train_loader, model, criterion, optimizer):
#     model.train()
#
#     # loss = 1
#     for step, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(args.device), targets.to(args.device)
#         optimizer.zero_grad()  #梯度归零
#         outputs = model(inputs)  #模型输出（分类情况）
#         loss = criterion(outputs, targets)  #根据真实类别和模型划分的类别情况计算损失
#         loss.backward()  #反向传播计算得到每个参数的梯度值
#         optimizer.step()  #通过梯度下降执行一步参数更新
#     # print(f"loss={loss.item():.4f}")

def train_epoch(args, train_loader, model, criterion, optimizer, epoch):
    model.train()

    # loss = 1
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()  #梯度归零
        outputs = model(inputs)  #模型输出（分类情况）
        loss = criterion(outputs, targets)  #根据真实类别和模型划分的类别情况计算损失
        loss.backward()  #反向传播计算得到每个参数的梯度值
        optimizer.step()  #通过梯度下降执行一步参数更新
    # print(f"loss={loss.item():.4f}")
        # 只在第一个 step 打印 FLOPs 和显存信息
        # if step == 0 and epoch ==199:
        #     # FLOPs
        #     flops = FlopCountAnalysis(model, inputs)
        #     total_flops = flops.total()
        #     print(f"[FLOPs] Epoch {epoch+1}: {total_flops / 1e9:.4f} GFLOPs")
        #     # 显存使用情况
        #     if torch.cuda.is_available():
        #         allocated = torch.cuda.memory_allocated(args.device) / (1024 ** 2)
        #         reserved = torch.cuda.memory_reserved(args.device) / (1024 ** 2)
        #         print(f"[GPU MEM] Epoch {epoch + 1} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

#我们还需要一个总的训练函数，一次调用就代表我们实验重复一次
def train(args, iter, model, train_loader, val_loader, criterion, optimizer, scheduler, epoch_num):
    start = time.time()
    best_val_OA = 0.0   #测试集最好的总体分类精度
    best_model = None   #记录性能最好的模型的权重

    for epoch in range(epoch_num):
        # Choice Model Training
        print('-------Iter:{}....train weights for epoch:{}-------'.format(iter, epoch + 1))
        train_epoch(args, train_loader, model, criterion, optimizer, epoch)
        #更新学习率计划
        scheduler.step()
        # Choice Model Validation 验证模型
        val_OA = validate(model, val_loader)  #验证集的总体分类精度
        print('val weights for epoch:{},val_OA={}'.format(epoch + 1, val_OA))
        # Save Best Model Weights  保存最佳模型权重参数
        if best_val_OA <= val_OA:
            best_val_OA = val_OA
            best_model = model
            print('Best val_OA by now,update the best model')

    print('*****Iter:{}....This iteration has ended, best_val_OA:{}, , return the best model*****'.format(iter, best_val_OA))
    # Record Time
    time_record(start)
    #这里只需要返回我们训练的最好权重即可
    return best_model
