import os
import sys
import glob
import time

import cv2
import torch

from tqdm import tqdm
from torch import einsum
import dataLoader as DLr
from utils import consistency
from network import MyNetwork
from utils.cuda_chk import GPUorCPU
from torch.utils.data import DataLoader

class Fusion:
    def __init__(self,
                modelpath='./pre-trained weights/best_network.pth',   # 训练好的权重位置
                dataroot='./Datasets/Eval',     # 测试数据集根目录
                dataset_name='Lytro/Triple Series',           # 数据集名称
                threshold=0.005,                # 小区域移除阈值
                ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.SAVEPATH = '/' + self.DATASET_NAME
        self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME == 'Lytro' or 'MFFW' or 'MFI' or 'Grayscale' or 'Lytro/Triple Series':
            # 加载已训练好的权重
            MODEL = self.LoadWeights(self.MODELPATH)
            # 准备数据
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            # 图像融合
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            # 为正确选择测试数据集
            print("Test Dataset required!")
            pass

    # 加载已训练好的模型权重
    def LoadWeights(self, modelpath):
        # 实例化模型到计算设备
        model = MyNetwork().to(self.DEVICE)
        # 总文件加载权重
        model.load_state_dict(torch.load(modelpath, map_location=torch.device(self.DEVICE)))
        # 将模型转为评估模式
        model.eval()
        # 打印模型相关信息
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
        return model

    # 准备数据
    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))   # 待测源图像A
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))   # 待测源图像B
        return eval_list_A, eval_list_B

    # 一致性检验
    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = consistency.Binarization(img_tensor)  # 二值化
        if threshold != 0:
            # 小区域移除
            Verified_img_tensor = consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results' + savepath):
            # 结果输出目录
            os.mkdir('./Results' + savepath)
        # 配置DataLoader
        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,     # 已配置的Dataloader
                                 batch_size=1,          # 批大小
                                 shuffle=False)         # 是否随机洗牌（评估时建议关闭）
        # 实例化tqdm对象
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                # 记录开始时间
                start_time = time.time()
                # 源图像喂模型，进行预测
                NetOut = model(A, B)
                # 对网络的预测结果进行一致性检验得到融合决策图D
                D = self.ConsisVerif(NetOut, threshold)[0]
                #################################### Image fusion #######################################
                D = einsum('c w h -> w h c', D).clone().detach().cpu().numpy()  # 转为opencv支持的whc模式
                A = cv2.imread(eval_list_A[cnt - 1])                            # 用opencv打开源图像A
                B = cv2.imread(eval_list_B[cnt - 1])                            # 用opencv打开源图像B
                Final_fused = A * D + B * (1 - D)                               # 图像融合
                # 写回聚焦决策图和融合图像
                cv2.imwrite('./Results' + savepath + '/lytro-' + str(cnt).zfill(2) + '-dm.png', D*255)
                cv2.imwrite('./Results' + savepath + '/lytro-' + str(cnt).zfill(2) + '.png', Final_fused)
                #########################################################################################
                # 记录每张图的融合时间
                running_time.append(time.time() - start_time)
                cnt += 1

        # 打印融合时间
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()

