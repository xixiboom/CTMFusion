
import os
import sys
import glob
import time
import numpy as np
import cv2
import torch

from tqdm import tqdm
import dataLoader as DLr
from utils import consistency
from nets.network import MyNetwork
from utils.cuda_chk import GPUorCPU
from torch.utils.data import DataLoader

# 强制使用 PyTorch 官方的 FLOPs 计数器
try:
    from torch.utils.flop_counter import FlopCounterMode

    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False
    print("Warning: torch.utils.flop_counter not available. Please upgrade PyTorch to >= 2.0")


class Fusion:
    def __init__(self,
                 modelpath='./pre-trained weights/best_network.pth',
                 dataroot='./Datasets/Eval',
                 dataset_name='Lytro',
                 threshold=0.005,
                 ):
        self.DEVICE = torch.device(GPUorCPU().DEVICE)
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.SAVEPATH = '/' + self.DATASET_NAME
        self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME

    def __call__(self, *args, **kwargs):
        MODEL = self.LoadWeights(self.MODELPATH)
        EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
        self.EvaluateComplexity(MODEL, len(EVAL_LIST_A))
        self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)

    def LoadWeights(self, modelpath):
        # ★ 关键修复1：显式指定使用 DTCWT
        model = MyNetwork(wavelet_backend="dtcwt").to(self.DEVICE)

        state_dict = torch.load(modelpath, map_location=self.DEVICE)

        # ★ 关键修复2：不再过滤 DTCWT buffer，完整加载
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warn] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[Warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"The number of model parameters: {num_params / 1e6:.4f} M\n")
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        Verified_img_tensor = consistency.Binarization(img_tensor)
        if threshold != 0:
            Verified_img_tensor = consistency.RemoveSmallArea(img_tensor=Verified_img_tensor, threshold=threshold)
        return Verified_img_tensor

    def EvaluateComplexity(self, model, num_pairs=4):
        print("=" * 80)
        print("Model Complexity and Runtime Evaluation")
        print("=" * 80)

        input_shape = (1, 3, 520, 520)
        print(f"{'Device':<30}: {self.DEVICE}")
        print(f"{'Dataset':<30}: {self.DATASET_NAME}")
        print(f"{'FLOPs input shape A/B':<30}: A: {list(input_shape)}, 'B': {list(input_shape)}")
        print(f"{'Precision for timing':<30}: FP32")
        print(f"{'Time mode':<30}: model(A, B), excluding data loading and saving")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        weight_size_mb = (param_size + buffer_size) / (1024 * 1024)

        print("-" * 80)
        print(f"{'Size (M)':<30}: {total_params / 1e6:.6f}")
        print(f"{'Trainable Params (M)':<30}: {trainable_params / 1e6:.6f}")
        print(f"{'Weight file size (MB)':<30}: {weight_size_mb:.3f}")

        A = torch.randn(*input_shape).to(self.DEVICE)
        B = torch.randn(*input_shape).to(self.DEVICE)

        if HAS_FLOP_COUNTER:
            flop_counter = FlopCounterMode(display=False)
            with torch.no_grad(), flop_counter:
                _ = model(A, B)
            total_flops = flop_counter.get_total_flops()
            print(f"{'FLOPs (G)':<30}: {total_flops / 1e9:.6f}")
            print(f"{'FLOPs backend':<30}: torch.utils.flop_counter")
        else:
            print(f"{'FLOPs (G)':<30}: N/A")

                # warmup 从 10 次增加到 30 次，排除首次编译和冷启动开销
        warmup_iters = 30
        bench_iters = 100
        discard_first = 10   # ★ 丢弃前 10 次，只统计后 90 次

        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(A, B)
            if self.DEVICE.type == 'cuda':
                torch.cuda.synchronize()

            times = []
            for _ in range(bench_iters):
                t0 = time.time()
                _ = model(A, B)
                if self.DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - t0)

        # ★ 关键：丢弃前 10 次，避免残留的编译/缓存抖动
        times = np.array(times[discard_first:])
        mean_time = float(np.mean(times))
        median_time = float(np.median(times))
        p95_time = float(np.percentile(times, 95))
        std_time = float(np.std(times))

        print(f"{'Times (s), mean':<30}: {mean_time:.6f}")
        print(f"{'Times (s), median':<30}: {median_time:.6f}")
        print(f"{'Times (s), p95':<30}: {p95_time:.6f}")
        print(f"{'Times (s), std':<30}: {std_time:.6f}")
        print(f"{'Number of image pairs':<30}: {num_pairs}")
        print("=" * 80)
        print("说明: FLOPs 取决于输入尺寸和统计工具。")
        print("=" * 80 + "\n")

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results' + savepath):
            os.mkdir('./Results' + savepath)

        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data, batch_size=1, shuffle=False)
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)

        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()
                NetOut = model(A, B)
                D = self.ConsisVerif(NetOut, threshold)[0]

                D = D.detach().float().cpu()
                D = torch.nan_to_num(D, nan=0.0, posinf=1.0, neginf=0.0)
                D = D.clamp(0.0, 1.0)

                D = torch.einsum('c w h -> w h c', D).contiguous().numpy().astype(np.float32)

                if D.ndim == 2:
                    D = D[..., None]
                if D.shape[-1] == 1:
                    D = np.repeat(D, 3, axis=-1)

                A_img = cv2.imread(eval_list_A[cnt - 1]).astype(np.float32)
                B_img = cv2.imread(eval_list_B[cnt - 1]).astype(np.float32)

                if D.shape[0] != A_img.shape[0] or D.shape[1] != A_img.shape[1]:
                    D = cv2.resize(D, (A_img.shape[1], A_img.shape[0]), interpolation=cv2.INTER_LINEAR)

                Final_fused = A_img * D + B_img * (1.0 - D)
                Final_fused = np.clip(Final_fused, 0, 255).astype(np.uint8)

                D_vis = np.clip(D[..., 0] * 255.0, 0, 255).astype(np.uint8)

                cv2.imwrite('./Results' + savepath + '/lytro-' + str(cnt).zfill(2) + '.png', Final_fused)

                running_time.append(time.time() - start_time)
                cnt += 1

        for t in running_time:
            print("process_time: {} s".format(t))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()