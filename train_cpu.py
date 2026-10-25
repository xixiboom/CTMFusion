import os
import glob
import sys
import datetime
import numpy as np

import torch
from torch import nn
from tqdm import tqdm
import dataLoader as DLr
from nets.network import MyNetwork
from utils.cuda_chk import GPUorCPU
from torch.utils.data import DataLoader
from utils.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F


class NetTrain:
    # 超参数
    def __init__(self,
                 data_path='./Datasets/Train&Valid/MFFdatasets',  # 训练数据集根目录
                 set_size=7866,  # 用多少数据
                 batchsize=8,  # 批大小
                 epochs=200,  # epoch总数
                 lr=0.0005,  # 初始学习率
                 gamma=0.88,  # 调度器迭代幅度
                 scheduler_step=1,  # 调度器迭代频率
                 patience=6,  # 早停阈值
                 save_dir='./results'  # 结果保存目录
                 ):
        # Hyper parameters
        self.DEVICE = GPUorCPU().DEVICE  # 选择计算设备
        self.DATAPATH = data_path
        self.SETSIEZE = set_size
        self.BATCHSIZE = batchsize
        self.EPOCHS = epochs
        self.LR = lr
        self.GAMMA = gamma
        self.SCHEDULER_STEP = scheduler_step
        self.PATIENCE = patience
        self.SAVE_DIR = save_dir

        # 创建保存目录
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.SAVE_DIR, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.SAVE_DIR, 'logs'), exist_ok=True)

        # 训练历史记录
        self.train_history = {
            'loss': [], 'accuracy': [], 'ssim': [], 'psnr': [],
            'val_loss': [], 'val_accuracy': [], 'val_ssim': [], 'val_psnr': []
        }

        # 日志文件
        self.log_file = open(os.path.join(self.SAVE_DIR, 'logs', 'training_log.txt'), 'w')

        # 初始化最佳指标
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0
        self.best_val_ssim = 0

    # Magicmethod
    def __call__(self, *args, **kwargs):
        try:
            # 准备数据
            TRAIN_LOADER, VALID_LOADER = self.PrepareDataLoader(self.DATAPATH, self.SETSIEZE, self.BATCHSIZE)
            # 构建训练要素
            MODEL, OPTIMIZER, SCHEDULER = self.BuildModel(self.DEVICE, self.LR, self.SCHEDULER_STEP, self.GAMMA)
            # 正式训练模型
            self.TrainingProcess(MODEL, OPTIMIZER, SCHEDULER, TRAIN_LOADER, VALID_LOADER, self.EPOCHS)
        except Exception as e:
            print(f"Training failed with error: {e}")
        finally:
            self.log_file.close()

    # 准备数据集（训练、验证）
    def PrepareDataLoader(self, datapath, setsize, batchsize):
        # 读取数据集文件
        train_list_A = sorted(glob.glob(os.path.join(datapath, 'train/sourceA', '*.*')))[:setsize]
        train_list_B = sorted(glob.glob(os.path.join(datapath, 'train/sourceB', '*.*')))[:setsize]
        train_list_GT = sorted(glob.glob(os.path.join(datapath, 'train/decisionmap', '*.*')))[:setsize]
        valid_list_A = sorted(glob.glob(os.path.join(datapath, 'validate/sourceA', '*.*')))[:setsize // 9]
        valid_list_B = sorted(glob.glob(os.path.join(datapath, 'validate/sourceB', '*.*')))[:setsize // 9]
        valid_list_GT = sorted(glob.glob(os.path.join(datapath, 'validate/decisionmap', '*.*')))[:setsize // 9]

        print(f"Train Data A: {len(train_list_A)}")
        print(f"Train Data B: {len(train_list_B)}")
        print(f"Train Data GT: {len(train_list_GT)}\n")
        print(f"Valid Data A: {len(valid_list_A)}")
        print(f"Valid Data B: {len(valid_list_B)}")
        print(f"Valid Data GT: {len(valid_list_GT)}\n")

        # 配置Dataloader
        train_data = DLr.DataLoader_Train(train_list_A, train_list_B, train_list_GT)
        valid_data = DLr.DataLoader_Train(valid_list_A, valid_list_B, valid_list_GT)

        # 实例化Dataloader
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
        valid_loader = DataLoader(dataset=valid_data,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=False)

        print(
            f"Train Data Size:{len(train_data)} , Train Loader Amount: {len(train_data)}/{batchsize} = {len(train_loader)}")
        print(
            f"Valid Data Size:{len(valid_data)} , Valid Loader Amount: {len(valid_data)}/{batchsize} = {len(valid_loader)}\n")

        return train_loader, valid_loader

    # 构建模型
    def BuildModel(self, device, lr, scheduler_step, gamma):
        model = MyNetwork().to(device)  # 实例化模型
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 添加权重衰减防止过拟合
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=gamma)

        # 显示模型参数量
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        params_mb = num_params / 1e6
        print(f"\n{'=' * 60}")
        print(f"Model Parameters: {params_mb:.2f} M")
        print(f"Model Layers: {len(list(model.modules()))}")
        print(f"{'=' * 60}\n")

        return model, optimizer, scheduler

    # 计算PSNR
    def calculate_psnr(self, img1, img2, max_val=1.0):
        """
        计算峰值信噪比 (PSNR)
        Args:
            img1, img2: 输入图像 (0-1范围)
            max_val: 最大像素值
        Returns:
            PSNR值
        """
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()

    # 计算SSIM (简化版本)
    def calculate_ssim(self, img1, img2, window_size=11):
        """
        计算结构相似性指数 (SSIM)
        Args:
            img1, img2: 输入图像 (0-1范围)
            window_size: 滑动窗口大小
        Returns:
            SSIM值
        """
        # 转换为numpy
        if torch.is_tensor(img1):
            img1 = img1.cpu().detach().numpy()
            img2 = img2.cpu().detach().numpy()

        # 确保是2D图像 (如果是批量，取第一个)
        if img1.ndim == 4:
            img1 = img1[0, 0] if img1.shape[1] == 1 else img1[0]
            img2 = img2[0, 0] if img2.shape[1] == 1 else img2[0]
        elif img1.ndim == 3 and img1.shape[0] == 1:
            img1 = img1[0]
            img2 = img2[0]

        # 计算SSIM
        try:
            ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min())
        except:
            ssim_value = 0.0

        return ssim_value

    # 计算准确率 (对于回归任务，使用1 - 归一化损失作为准确率)
    def calculate_accuracy(self, pred, target):
        """
        计算准确率 (对于图像融合任务)
        Args:
            pred: 预测图像
            target: 真实图像
        Returns:
            准确率
        """
        # 方法1: 基于像素误差的准确率
        error = torch.abs(pred - target)
        accuracy = 1.0 - torch.mean(error)

        # 方法2: 基于阈值化的准确率 (如果任务需要二分类)
        # threshold = 0.5
        # pred_binary = (pred > threshold).float()
        # target_binary = (target > threshold).float()
        # accuracy = (pred_binary == target_binary).float().mean()

        return accuracy.item()

    # 构造损失函数
    def LossFunc(self, GT, NetOut):
        # 确保GT和NetOut都是张量
        if isinstance(GT, tuple):
            for item in GT:
                if torch.is_tensor(item):
                    GT = item
                    break
            else:
                raise ValueError("GT is tuple but contains no tensor")

        # 确保形状匹配
        if GT.dim() == 4 and NetOut.dim() == 3:
            NetOut = NetOut.unsqueeze(1)

        # 计算L1损失
        loss_abs = nn.L1Loss()
        loss = loss_abs(GT, NetOut)

        return loss

    # 打印模型评估结果
    def print_evaluation(self, epoch, train_metrics, val_metrics, learning_rate):
        """
        打印详细的评估结果
        Args:
            epoch: 当前epoch
            train_metrics: 训练集指标字典
            val_metrics: 验证集指标字典
            learning_rate: 当前学习率
        """
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch:03d} - Learning Rate: {learning_rate:.6f}")
        print(f"{'=' * 80}")
        print(f"{'Metric':<15} {'Train':<15} {'Validation':<15} {'Difference':<15}")
        print(f"{'-' * 80}")
        print(f"{'Loss':<15} {train_metrics['loss']:.6f}        {val_metrics['loss']:.6f}        "
              f"{train_metrics['loss'] - val_metrics['loss']:.6f}")
        print(f"{'Accuracy':<15} {train_metrics['accuracy']:.4f}        {val_metrics['accuracy']:.4f}        "
              f"{train_metrics['accuracy'] - val_metrics['accuracy']:.4f}")
        print(f"{'PSNR':<15} {train_metrics['psnr']:.2f} dB      {val_metrics['psnr']:.2f} dB      "
              f"{train_metrics['psnr'] - val_metrics['psnr']:.2f} dB")
        print(f"{'SSIM':<15} {train_metrics['ssim']:.4f}        {val_metrics['ssim']:.4f}        "
              f"{train_metrics['ssim'] - val_metrics['ssim']:.4f}")

        # 判断是否是最佳模型
        if val_metrics['loss'] < self.best_val_loss:
            print(f"✓ New best validation loss: {val_metrics['loss']:.6f} (previous: {self.best_val_loss:.6f})")
            self.best_val_loss = val_metrics['loss']

        if val_metrics['psnr'] > self.best_val_psnr:
            print(f"✓ New best PSNR: {val_metrics['psnr']:.2f} dB (previous: {self.best_val_psnr:.2f} dB)")
            self.best_val_psnr = val_metrics['psnr']

        if val_metrics['ssim'] > self.best_val_ssim:
            print(f"✓ New best SSIM: {val_metrics['ssim']:.4f} (previous: {self.best_val_ssim:.4f})")
            self.best_val_ssim = val_metrics['ssim']

        print(f"{'=' * 80}\n")

    # 保存训练曲线
    def save_training_curves(self):
        """保存训练过程中的指标曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss曲线
        axes[0, 0].plot(self.train_history['loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy曲线
        axes[0, 1].plot(self.train_history['accuracy'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.train_history['val_accuracy'], label='Val Accuracy', color='red')
        axes[0, 1].set_title('Accuracy Curve')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # PSNR曲线
        axes[1, 0].plot(self.train_history['psnr'], label='Train PSNR', color='blue')
        axes[1, 0].plot(self.train_history['val_psnr'], label='Val PSNR', color='red')
        axes[1, 0].set_title('PSNR Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # SSIM曲线
        axes[1, 1].plot(self.train_history['ssim'], label='Train SSIM', color='blue')
        axes[1, 1].plot(self.train_history['val_ssim'], label='Val SSIM', color='red')
        axes[1, 1].set_title('SSIM Curve')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.SAVE_DIR, 'logs', 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 模型训练过程
    def TrainingProcess(self, model, optimizer, scheduler, train_loader, valid_loader, epochs):
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        early_stopping = EarlyStopping(self.SAVE_DIR, patience=self.PATIENCE, verbose=True)

        print('\n' + '=' * 60)
        print('Training Start!')
        print(f'Device: {self.DEVICE}')
        print(f'Train samples: {len(train_loader.dataset)}')
        print(f'Valid samples: {len(valid_loader.dataset)}')
        print(f'Batch size: {self.BATCHSIZE}')
        print(f'Epochs: {epochs}')
        print('=' * 60 + '\n')

        for epoch in range(epochs):
            ######################################### Train #########################################
            model.train()
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            epoch_train_psnr = 0
            epoch_train_ssim = 0
            train_batches = 0

            train_loader_tqdm = tqdm(train_loader, desc=f'[Train] Epoch {epoch + 1}/{epochs}',
                                     colour='green', leave=False, file=sys.stdout)

            for A, B, GT in train_loader_tqdm:
                # 移动到设备
                A = A.to(self.DEVICE)
                B = B.to(self.DEVICE)
                GT = GT.to(self.DEVICE)

                optimizer.zero_grad()

                # 混合精度训练 (如果有GPU)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        NetOut = model(A, B)
                        loss = self.LossFunc(GT, NetOut)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    NetOut = model(A, B)
                    loss = self.LossFunc(GT, NetOut)
                    loss.backward()
                    optimizer.step()

                # 计算指标
                accuracy = self.calculate_accuracy(NetOut, GT)
                psnr = self.calculate_psnr(NetOut, GT)
                ssim_value = self.calculate_ssim(NetOut, GT)

                # 累积指标
                epoch_train_loss += loss.item()
                epoch_train_accuracy += accuracy
                epoch_train_psnr += psnr
                epoch_train_ssim += ssim_value
                train_batches += 1

                # 更新进度条
                train_loader_tqdm.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}',
                    'psnr': f'{psnr:.2f}dB'
                })

            # 计算平均指标
            avg_train_loss = epoch_train_loss / train_batches
            avg_train_accuracy = epoch_train_accuracy / train_batches
            avg_train_psnr = epoch_train_psnr / train_batches
            avg_train_ssim = epoch_train_ssim / train_batches

            ######################################### Valid #########################################
            model.eval()
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            epoch_val_psnr = 0
            epoch_val_ssim = 0
            val_batches = 0

            with torch.no_grad():
                valid_loader_tqdm = tqdm(valid_loader, desc=f'[Valid] Epoch {epoch + 1}/{epochs}',
                                         colour='yellow', leave=False, file=sys.stdout)

                for A, B, GT in valid_loader_tqdm:
                    A = A.to(self.DEVICE)
                    B = B.to(self.DEVICE)
                    GT = GT.to(self.DEVICE)

                    NetOut = model(A, B)
                    loss = self.LossFunc(GT, NetOut)

                    # 计算指标
                    accuracy = self.calculate_accuracy(NetOut, GT)
                    psnr = self.calculate_psnr(NetOut, GT)
                    ssim_value = self.calculate_ssim(NetOut, GT)

                    # 累积指标
                    epoch_val_loss += loss.item()
                    epoch_val_accuracy += accuracy
                    epoch_val_psnr += psnr
                    epoch_val_ssim += ssim_value
                    val_batches += 1

                    # 更新进度条
                    valid_loader_tqdm.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{accuracy:.4f}',
                        'psnr': f'{psnr:.2f}dB'
                    })

            # 计算平均指标
            avg_val_loss = epoch_val_loss / val_batches
            avg_val_accuracy = epoch_val_accuracy / val_batches
            avg_val_psnr = epoch_val_psnr / val_batches
            avg_val_ssim = epoch_val_ssim / val_batches

            ######################################### 保存记录 #########################################
            # 保存到历史记录
            self.train_history['loss'].append(avg_train_loss)
            self.train_history['accuracy'].append(avg_train_accuracy)
            self.train_history['psnr'].append(avg_train_psnr)
            self.train_history['ssim'].append(avg_train_ssim)
            self.train_history['val_loss'].append(avg_val_loss)
            self.train_history['val_accuracy'].append(avg_val_accuracy)
            self.train_history['val_psnr'].append(avg_val_psnr)
            self.train_history['val_ssim'].append(avg_val_ssim)

            # 准备指标字典
            train_metrics = {
                'loss': avg_train_loss,
                'accuracy': avg_train_accuracy,
                'psnr': avg_train_psnr,
                'ssim': avg_train_ssim
            }

            val_metrics = {
                'loss': avg_val_loss,
                'accuracy': avg_val_accuracy,
                'psnr': avg_val_psnr,
                'ssim': avg_val_ssim
            }

            # 打印详细评估结果
            self.print_evaluation(epoch + 1, train_metrics, val_metrics, scheduler.get_last_lr()[0])

            # 保存日志到文件
            log_line = f"Epoch {epoch + 1:03d}, Train Loss: {avg_train_loss:.6f}, Train Acc: {avg_train_accuracy:.4f}, "
            log_line += f"Train PSNR: {avg_train_psnr:.2f}dB, Train SSIM: {avg_train_ssim:.4f}, "
            log_line += f"Val Loss: {avg_val_loss:.6f}, Val Acc: {avg_val_accuracy:.4f}, "
            log_line += f"Val PSNR: {avg_val_psnr:.2f}dB, Val SSIM: {avg_val_ssim:.4f}, "
            log_line += f"LR: {scheduler.get_last_lr()[0]:.6f}\n"

            self.log_file.write(log_line)
            self.log_file.flush()

            # 动态学习率调整
            scheduler.step()

            # 保存最佳模型
            if avg_val_loss < self.best_val_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_psnr': avg_val_psnr,
                    'val_ssim': avg_val_ssim,
                    'train_history': self.train_history
                }, os.path.join(self.SAVE_DIR, 'checkpoints', 'best_model.pth'))

                print(f"✓ Best model saved with val_loss: {avg_val_loss:.6f}, val_psnr: {avg_val_psnr:.2f}dB\n")

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, os.path.join(self.SAVE_DIR, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth'))

            # 早停检查
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        ######################################### 训练结束 #########################################
        print('\n' + '=' * 60)
        print('Training Complete!')
        print('=' * 60)

        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(self.SAVE_DIR, 'checkpoints', 'final_model.pth'))

        # 保存训练曲线
        self.save_training_curves()

        # 打印最佳指标
        print(f"\nBest Validation Metrics:")
        print(f"  Loss:    {self.best_val_loss:.6f}")
        print(f"  PSNR:    {self.best_val_psnr:.2f} dB")
        print(f"  SSIM:    {self.best_val_ssim:.4f}")
        print(f"  Accuracy: {max(self.train_history['val_accuracy']):.4f}")

        # 保存训练总结
        summary_file = os.path.join(self.SAVE_DIR, 'logs', 'training_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Training Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Epochs: {len(self.train_history['loss'])}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.6f}\n")
            f.write(f"Best Validation PSNR: {self.best_val_psnr:.2f} dB\n")
            f.write(f"Best Validation SSIM: {self.best_val_ssim:.4f}\n")
            f.write(f"Best Validation Accuracy: {max(self.train_history['val_accuracy']):.4f}\n")
            f.write(f"\nFinal Metrics:\n")
            f.write(f"  Train Loss: {self.train_history['loss'][-1]:.6f}\n")
            f.write(f"  Train PSNR: {self.train_history['psnr'][-1]:.2f} dB\n")
            f.write(f"  Val Loss: {self.train_history['val_loss'][-1]:.6f}\n")
            f.write(f"  Val PSNR: {self.train_history['val_psnr'][-1]:.2f} dB\n")

        print(f"\nTraining results saved to: {self.SAVE_DIR}")
        print(f"Logs saved to: {os.path.join(self.SAVE_DIR, 'logs')}")
        print(f"Checkpoints saved to: {os.path.join(self.SAVE_DIR, 'checkpoints')}")


if __name__ == '__main__':
    # 训练配置
    config = {
        'data_path': './Datasets/Train&Valid/MFFdatasets',
        'set_size': 7866,
        'batchsize': 8,
        'epochs': 200,
        'lr': 0.0005,
        'gamma': 0.88,
        'scheduler_step': 1,
        'patience': 10,
        'save_dir': f'./results/experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }

    print("\n" + "=" * 60)
    print("Starting Image Fusion Training")
    print("=" * 60)

    t = NetTrain(**config)
    t()