import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils.cuda_chk import GPUorCPU
from torchvision.io import read_image, ImageReadMode


DEVICE = GPUorCPU().DEVICE
model_input_image_size_height = 256
model_input_image_size_width = 256
random_crop_size = 224

# 归一化（除以255）
class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)     # float32 单精度

# 定义“训练/验证”数据加载器
class DataLoader_Train(Dataset):
    # 定义数据增强模块
    train_valid_transforms = transforms.Compose(
        [
            # transforms.CenterCrop(224),               # 中心裁剪
            transforms.Resize((model_input_image_size_height, model_input_image_size_width), antialias=False),  # 调整大小
            transforms.RandomCrop(random_crop_size),    # 随机裁剪
            transforms.RandomHorizontalFlip(),          # 随机水平反转
            transforms.RandomVerticalFlip(),            # 随机垂直反转
            ZeroOneNormalize(),                         # 数据归一化
        ]
    )
    # 定义数据标准化模块
    train_valid_transforms_Norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # 数据标准化，即均值为0，标准差为1
        ]
    )

    def __init__(self, file_list_A, file_list_B, file_list_GT):
        self.file_list_A = file_list_A                      # 打开源图像A
        self.file_list_B = file_list_B                      # 打开源图像B
        self.file_list_GT = file_list_GT                    # 打开真实决策图
        self.transform1 = self.train_valid_transforms       # 数据增强模块
        self.transform2 = self.train_valid_transforms_Norm  # 数据标准化模块

    # 返回数据集长度
    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B) == len(self.file_list_GT):
            self.filelength = len(self.file_list_A)
            return self.filelength

    # 返回图像个体
    def __getitem__(self, idx):
        seed = torch.random.seed()  # 固定随机种子

        imgA_path = self.file_list_A[idx]   # 源图像A地址列表
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)    # 以RGB模式读取源图像A
        torch.random.manual_seed(seed)  # 超控随机种子
        img_A = self.transform1(img_A)  # 执行数据增强
        imgA_transformed = self.transform2(img_A)   # 执行数据标准化

        imgB_path = self.file_list_B[idx]   # 源图像B地址列表
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)    # 以RGB模式读取源图像B
        torch.random.manual_seed(seed)  # 超控随机种子
        img_B = self.transform1(img_B)  # 执行数据增强
        imgB_transformed = self.transform2(img_B)   # 执行数据标准化

        imgGT_path = self.file_list_GT[idx] # 真实决策图地址列表
        img_GT = read_image(imgGT_path, mode=ImageReadMode.GRAY).to(DEVICE) # 以灰度模式读取真实决策图
        torch.random.manual_seed(seed)  # 超控随机种子
        imgGT_transformed = self.transform1(img_GT) # 执行数据增强

        return imgA_transformed, imgB_transformed, imgGT_transformed


# 定义“测试”数据加载器
class Dataloader_Eval(Dataset):
    eval_transforms = transforms.Compose(
        [
            # transforms.Resize((520, 520)),
            # transforms.ToTensor(),    # img对象转tensor
            ZeroOneNormalize(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(self, file_list_A, file_list_B):
        self.file_list_A = file_list_A
        self.file_list_B = file_list_B
        self.transform1 = self.eval_transforms
        self.transform2 = self.eval_transforms

    def __len__(self):
        if len(self.file_list_A) == len(self.file_list_B):
            self.filelength = len(self.file_list_A)
            return self.filelength

    def __getitem__(self, idx):
        imgA_path = self.file_list_A[idx]
        img_A = read_image(imgA_path, mode=ImageReadMode.RGB).to(DEVICE)
        imgA_transformed = self.transform1(img_A)

        imgB_path = self.file_list_B[idx]
        img_B = read_image(imgB_path, mode=ImageReadMode.RGB).to(DEVICE)
        imgB_transformed = self.transform1(img_B)

        return imgA_transformed, imgB_transformed
