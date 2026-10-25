import os
import glob
import sys
import datetime

import torch
from torch import nn
from tqdm import tqdm
import dataLoader as DLr
from nets.network import MyNetwork
from utils.cuda_chk import GPUorCPU
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping

class NetTrain:
    # 超参数
    def __init__(self,
                 data_path='./Datasets/Train&Valid/MFFdatasets',    # 训练数据集根目录
                 set_size=7866,     # 用多少数据
                 batchsize=8,       # 批大小
                 epochs=200,        # epoch总数
                 lr=0.0005,         # 初始学习率
                 gamma=0.88,        # 调度器迭代幅度
                 scheduler_step=1,  # 调度器迭代频率
                 patience=6         # 早停阈值
                 ):
        # Hyper parameters
        self.DEVICE = GPUorCPU().DEVICE     # 选择计算设备
        self.DATAPATH = data_path
        self.SETSIEZE = set_size
        self.BATCHSIZE = batchsize
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.PATIENCE = patience

    # Magicmethod
    def __call__(self, *args, **kwargs):
        # 准备数据
        TRAIN_LOADER, VALID_LOADER = self.PrepareDataLoader(self.DATAPATH, self.SETSIEZE, self.BATCHSIZE)
        # 构建训练要素
        MODEL, OPTIMIZER, SCHEDULER = self.BuildModel(self.DEVICE, self.LR, self.SCHEDULER_STEP, self.GAMMA)
        # 正式训练模型
        self.TrainingProcess(MODEL, OPTIMIZER, SCHEDULER, TRAIN_LOADER, VALID_LOADER, self.EPOCHS)

    # 准备数据集（训练、验证）
    def PrepareDataLoader(self, datapath, setsize, batchsize):
        # 读取数据集文件
        train_list_A = sorted(glob.glob(os.path.join(datapath, 'train/sourceA', '*.*')))[:setsize]
        train_list_B = sorted(glob.glob(os.path.join(datapath, 'train/sourceB', '*.*')))[:setsize]
        train_list_GT = sorted(glob.glob(os.path.join(datapath, 'train/decisionmap', '*.*')))[:setsize]
        valid_list_A = sorted(glob.glob(os.path.join(datapath, 'validate/sourceA', '*.*')))[:setsize // 9]
        valid_list_B = sorted(glob.glob(os.path.join(datapath, 'validate/sourceB', '*.*')))[:setsize // 9]
        valid_list_GT = sorted(glob.glob(os.path.join(datapath, 'validate/decisionmap', '*.*')))[:setsize // 9]
        tqdm.write(f"Train Data A: {len(train_list_A)}")
        tqdm.write(f"Train Data B: {len(train_list_B)}")
        tqdm.write(f"Train Data GT: {len(train_list_GT)}\n")
        tqdm.write(f"Valid Data A: {len(valid_list_A)}")
        tqdm.write(f"Valid Data B: {len(valid_list_B)}")
        tqdm.write(f"Valid Data GT: {len(valid_list_GT)}\n")
        # 配置Dataloader
        train_data = DLr.DataLoader_Train(train_list_A, train_list_B, train_list_GT)
        valid_data = DLr.DataLoader_Train(valid_list_A, valid_list_B, valid_list_GT)
        # 实例化Dataloader
        train_loader = DataLoader(dataset=train_data,       # 已配置的Dataloader
                                  batch_size=batchsize,     # 批大小
                                  shuffle=True,             # 是否随机洗牌（训练时务必开启）
                                  num_workers=0,            # 装载器数量（大于0时使用多进程执行数据预装载，windows慎用）
                                  pin_memory=False)         # 是否以“锁页”方式存入内存
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        tqdm.write(f"Train Data Size:{len(train_data)} , Train Loader Amount: {len(train_data)}/{batchsize} = {len(train_loader)}")
        tqdm.write(f"Valid Data Size:{len(valid_data)} , Valid Loader Amount: {len(valid_data)}/{batchsize} = {len(valid_loader)}\n")
        return train_loader, valid_loader

    # 构建模型
    def BuildModel(self, device, lr, scheduler_step, gamma):
        model = MyNetwork().to(device)  # 实例化模型
        # torch.compile(model) # pytorch2以上
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)   # 实例化优化器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=gamma)    # 实例化学习调度器
        # 显示模型参数量
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model, optimizer, scheduler

    # 构造损失函数
    def LossFunc(self, GT, NetOut):
        loss_abs = nn.L1Loss()  # 实例化pyotrch内置L1，与 loss_abs = torch.mean(torch.abs(GT - NetOut)) 同义
        loss = loss_abs(GT, NetOut)
        return loss

    # 模型训练过程
    def TrainingProcess(self, model, optimizer, scheduler, train_loader, valid_loader, epochs):
        torch.backends.cudnn.benchmark = True   # cudnn库自动寻找优化路径
        scaler = torch.cuda.amp.GradScaler()    # 实例化梯度缩放，帮助混合精度计算
        early_stopping = EarlyStopping('./')    # 实例化早停算法函数

        tqdm.write('Training start...\n')

        for epoch in range(epochs):
            ######################################### Train #########################################
            epoch_loss = 0      # 记录每个训练epoch的总损失
            epoch_accuracy = 0  # 记录每个训练epoch的总精确度
            train_loader_tqdm = tqdm(train_loader, colour='green', leave=False, file=sys.stdout)    # 实例化一个tqdm对象
            for A, B, GT in train_loader_tqdm:  # 调用tqdm对象，实现训练进度可视化
                optimizer.zero_grad()   # 优化器梯度置0
                # Automatic mixed precision training.
                with torch.autocast(device_type=self.DEVICE, dtype=torch.float16):  # 混合精度训练模型并计算loss
                    NetOut = model(A, B)    # 数据喂到模型，模型返回预测值
                    loss = self.LossFunc(GT, NetOut)    # 与ground truth求loss
                scaler.scale(loss).backward()   # loss回传
                scaler.step(optimizer)          # 梯度反向传导
                scaler.update()                 # 梯度缩放更新
                ''''''''''''''''''''''''''' Epoch loss calculation & progress visualization '''''''''''''''''''''''''''
                epoch_loss += loss / len(train_loader)  # 一个epoch中的一次训练的loss
                epoch_accuracy += (1 - loss) / len(train_loader)    # 一个epoch中的一次训练的acc
                # 配置进度条，显示实时loss等数据
                train_loader_tqdm.set_description("[%s] Epoch %s" % (str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')), str(epoch + 1)))
                train_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))
                ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################

            ######################################### Valid #########################################
            with torch.no_grad():       # 关闭梯度计算
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                valid_loader_tqdm = tqdm(valid_loader, colour='yellow', leave=False, file=sys.stdout)
                for A, B, GT in valid_loader_tqdm:
                    NetOut = model(A, B)
                    loss = self.LossFunc(GT, NetOut)
                    ''''''''''''''''''''' Epoch loss calculation & progress visualization '''''''''''''''''''''
                    epoch_val_accuracy += (1 - loss) / len(valid_loader)
                    epoch_val_loss += loss / len(valid_loader)
                    valid_loader_tqdm.set_description("[Validating...] Epoch %s" % str(epoch + 1))
                    valid_loader_tqdm.set_postfix(loss=float(loss), acc=1 - float(loss))
                    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            #########################################################################################
            # Print epoch loss and accuracy.
            tqdm.write(f"[{str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S'))}] Epoch {epoch + 1} - "
                       f"loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - "
                       f"val_acc: {epoch_val_accuracy:.4f}")
            # Dynamic learning rate.
            scheduler.step()    # 调度器更新
            # 早停止
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
        # Epoch loop ends.


if __name__ == '__main__':
    t = NetTrain()
    t()
