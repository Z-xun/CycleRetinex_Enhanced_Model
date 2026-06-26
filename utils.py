from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import torch.nn as nn
from config import *
class MyDataset_paired(Dataset):
    def __init__(self, low_path, high_path, prefix_low='low', prefix_high='normal'):
        """
        配对数据集加载器
        
        Args:
            low_paths: 低光照图像文件夹路径列表 [路径1, 路径2]
            high_paths: 高光照图像文件夹路径列表 [路径1, 路径2]
            prefix_low: 低光照图像文件名前缀（如 'low'）
            prefix_high: 高光照图像文件名前缀（如 'norm'）
        """
        super().__init__()
        
        self.pairs = []
        
        low_imgs = sorted([
            os.path.join(low_path, p) for p in os.listdir(low_path) 
            if p.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        for low_img_path in low_imgs:
            basename = os.path.basename(low_img_path)
            # 替换前缀
            high_basename = basename.replace(prefix_low, prefix_high)
            # 构造高光图像的完整路径
            high_img_path = os.path.join(high_path, high_basename)
            
            # 检查高光图像是否存在
            if os.path.exists(high_img_path):
                self.pairs.append([low_img_path, high_img_path])
        # ====== 数据增强 ======
        self.transforms = A.Compose([
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        low_path,high_path= self.pairs[index]
        
        # 加载低光照图像
        img_low = Image.open(low_path).convert("RGB")
        augment_low = self.transforms(image=np.array(img_low))
        img_low_data = augment_low["image"].float() / 255.0
      
        img_high = Image.open(high_path).convert("RGB")
        augment_high = self.transforms(image=np.array(img_high))
        img_high_data = augment_high["image"].float() / 255.0
        
        return img_low_data, img_high_data


class MyDataset_unpaired(Dataset):
    def __init__(self, low_path, high_path):
        """
        配对数据集加载器
        
        Args:
            low_paths: 低光照图像文件夹路径列表 [路径1, 路径2]
            high_paths: 高光照图像文件夹路径列表 [路径1, 路径2]
            prefix_low: 低光照图像文件名前缀（如 'low'）
            prefix_high: 高光照图像文件名前缀（如 'norm'）
        """
        super().__init__()
        
        self.pairs = []
        
        self.low_imgs = sorted([
            os.path.join(low_path, p) for p in os.listdir(low_path) 
            if p.endswith(('.jpg', '.png', '.jpeg'))
        ])
        

        self.high_imgs = sorted([
            os.path.join(high_path, p) for p in os.listdir(high_path) 
            if p.endswith(('.jpg', '.png', '.jpeg'))
        ])


        self.high_len = len(self.high_imgs)
        self.low_len = len(self.low_imgs)
        self.transforms = A.Compose([
            ToTensorV2()
        ])
    
    def __len__(self):
        return min(self.high_len,self.low_len)
    
    def __getitem__(self, index):
        low_path = self.low_imgs[index]
        
        # 加载低光照图像
        img_low = Image.open(low_path).convert("RGB")
        augment_low = self.transforms(image=np.array(img_low))
        img_low_data = augment_low["image"].float() / 255.0
        
        high_idx = random.randint(0, self.high_len - 1)
        high_path = self.high_imgs[high_idx]
        # 加载配对的高光照图像
        img_high = Image.open(high_path).convert("RGB")
        augment_high = self.transforms(image=np.array(img_high))
        img_high_data = augment_high["image"].float() / 255.0
        
        return img_low_data, img_high_data

def save_checkpoint(epoch,
                    Decom_net,
                    L2H_net,
                    H2L_net,
                    Denoise_net,
                    D_L_low,
                    D_L_high,
                    D_R,
                    optimizer_Decom,
                    optimizer_L2H,
                    optimizer_H2L,
                    optimizer_Denoise,
                    optimizer_D_L_low,
                    optimizer_D_L_high,
                    optimizer_D_R,
                    loss_history,
                    metric_history=None,
                    best_metric_info=None,
                    save_dir=save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        # 训练信息
        'epoch': epoch,
        'lr': lr,
        'betas': betas,
        
        'Decom_net_state_dict': Decom_net.state_dict(),
        'Denoise_net_state_dict':Denoise_net.state_dict(),
        'L2H_net_state_dict': L2H_net.state_dict(),
        'H2L_net_state_dict': H2L_net.state_dict(),
        'D_L_low_state_dict': D_L_low.state_dict(),
        'D_L_high_state_dict': D_L_high.state_dict(),
        'D_R_state_dict': D_R.state_dict(),
    
        'optim_Denoise_net_state_dict':optimizer_Denoise.state_dict(),
        'optimizer_Decom_state_dict': optimizer_Decom.state_dict(),

        'optimizer_L2H_state_dict': optimizer_L2H.state_dict(),
        'optimizer_H2L_state_dict': optimizer_H2L.state_dict(),
        'optimizer_D_L_low_state_dict': optimizer_D_L_low.state_dict(),
        'optimizer_D_L_high_state_dict': optimizer_D_L_high.state_dict(),
        'optimizer_D_R_state_dict': optimizer_D_R.state_dict(),
        
        # 损失历史
        'loss_history': loss_history,
        'metric_history': metric_history or {},
        'best_metric_info': best_metric_info or {},
    }

    save_path = f'{save_dir}/checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, save_path)
    print(f" 检查点已保存：{save_path}")
    return save_path
def smooth_loss(L, img):
    """结构感知的平滑损失"""
    # 计算光照图 L 的梯度
    diff_h_L = L[:, :, :, 1:] - L[:, :, :, :-1]
    diff_w_L = L[:, :, 1:, :] - L[:, :, :-1, :]
    
    # 计算原图 img 的梯度
    diff_h_img = img[:, :, :, 1:] - img[:, :, :, :-1]
    diff_w_img = img[:, :, 1:, :] - img[:, :, :-1, :]
    
    # 计算权重：原图梯度越大的地方（边缘），权重越小（允许 L 存在自然跳变）
    weight_h = torch.exp(-10.0 * torch.mean(torch.abs(diff_h_img), dim=1, keepdim=True))
    weight_w = torch.exp(-10.0 * torch.mean(torch.abs(diff_w_img), dim=1, keepdim=True))
    
    loss_h = torch.mean(torch.abs(diff_h_L) * weight_h)
    loss_w = torch.mean(torch.abs(diff_w_L) * weight_w)
    
    return loss_h + loss_w

#模糊L分量图，让细节进入R图中
def average_blur(x, kernel_size=15):
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

def illumination_prior_map(img, kernel_size=15):
    max_rgb = torch.max(img, dim=1, keepdim=True)[0]
    return average_blur(max_rgb, kernel_size=kernel_size)

def illumination_statistics_loss(pred, target):
    pred_flat = pred.flatten(start_dim=2)
    target_flat = target.flatten(start_dim=2)

    pred_mean = pred_flat.mean(dim=2)
    target_mean = target_flat.mean(dim=2)
    pred_std = pred_flat.std(dim=2, unbiased=False)
    target_std = target_flat.std(dim=2, unbiased=False)

    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)

def illumination_prior_loss(L, img, kernel_size=15):
    target = illumination_prior_map(img, kernel_size=kernel_size)

    diff_h_L = L[:, :, :, 1:] - L[:, :, :, :-1]
    diff_w_L = L[:, :, 1:, :] - L[:, :, :-1, :]
    diff_h_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    diff_w_target = target[:, :, 1:, :] - target[:, :, :-1, :]

    value_loss = F.l1_loss(L, target)
    gradient_loss = F.l1_loss(diff_h_L, diff_h_target) + F.l1_loss(diff_w_L, diff_w_target)
    return value_loss + 0.5 * gradient_loss

def calculate_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse.item() == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2, device=pred.device) / mse).item()

def calculate_ssim_tensor(pred, target, max_val=1.0):
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12
    ssim_map = numerator / denominator
    return ssim_map.mean()

def calculate_ssim(pred, target, max_val=1.0):
    return calculate_ssim_tensor(pred, target, max_val=max_val).item()

def save_loss_plot(loss_history, save_dir):
    """绘制损失曲线图"""
    plt.figure(figsize=(15, 5))
    
    # 子图 1: 生成器损失
    plt.subplot(1, 3, 1)
    plt.plot(loss_history['G_total'], label='G Total', linewidth=2, color='red')
    plt.plot(loss_history['cycle'], label='Cycle', alpha=0.7)
    plt.plot(loss_history['G_L'], label='G_L', alpha=0.7)
    plt.plot(loss_history['G_R'], label='G_R', alpha=0.7)
    plt.plot(loss_history['recon'], label='Recon', alpha=0.7)
    if loss_history.get('illum_prior'):
        plt.plot(loss_history['illum_prior'], label='Illum Prior', alpha=0.7)
    if loss_history.get('illum_match'):
        plt.plot(loss_history['illum_match'], label='Illum Match', alpha=0.7)
    if loss_history.get('cross_recon'):
        plt.plot(loss_history['cross_recon'], label='Cross Recon', alpha=0.7)
    if loss_history.get('illum_stats'):
        plt.plot(loss_history['illum_stats'], label='Illum Stats', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 2: 判别器损失
    plt.subplot(1, 3, 2)
    plt.plot(loss_history['D_L_low'], label='D_L_low', linewidth=2, color='blue')
    plt.plot(loss_history['D_L_high'], label='D_L_high', alpha=0.7)
    plt.plot(loss_history['D_R'], label='D_R', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图 3: 总损失
    plt.subplot(1, 3, 3)
    plt.plot(loss_history['G_total'], label='G Total', color='red', linewidth=2)
    plt.plot(loss_history['D_total'], label='D Total', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_curve.png', dpi=150)
    plt.close()
    print(f" 损失曲线已保存至 {save_dir}/loss_curve.png")


def save_metric_plot(metric_history, save_dir):
    if not metric_history:
        return

    plt.figure(figsize=(10, 4))

    if metric_history.get("psnr"):
        plt.plot(metric_history["psnr"], label="PSNR", linewidth=2, color="green")
    if metric_history.get("ssim"):
        plt.plot(metric_history["ssim"], label="SSIM", linewidth=2, color="purple")
    if metric_history.get("score"):
        plt.plot(metric_history["score"], label="Score", linewidth=2, color="orange")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metric_curve.png", dpi=150)
    plt.close()
    print(f" éªŒè¯æŒ‡æ ‡æ›²çº¿å·²ä¿å­˜åˆ° {save_dir}/metric_curve.png")

def load_checkpoint(checkpoint_path,
                    Decom_net,
                    L2H_net,
                    H2L_net,
                    Denoise_net,
                    D_L_low,
                    D_L_high,
                    D_R,
                    optimizer_Denoise_net,
                    optimizer_Decom,
                    optimizer_L2H,
                    optimizer_H2L,
                    optimizer_D_L_low,
                    optimizer_D_L_high,
                    optimizer_D_R,
                    device=device):
    """加载模型检查点（恢复训练）"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在：{checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    Decom_net.load_state_dict(checkpoint['Decom_net_state_dict'])
    Denoise_net.load_state_dict(checkpoint['Denoise_net_state_dict'])
    L2H_net.load_state_dict(checkpoint['L2H_net_state_dict'])
    H2L_net.load_state_dict(checkpoint['H2L_net_state_dict'])
    D_L_low.load_state_dict(checkpoint['D_L_low_state_dict'])
    D_L_high.load_state_dict(checkpoint['D_L_high_state_dict'])
    D_R.load_state_dict(checkpoint['D_R_state_dict'])
    
    optimizer_Denoise_net.load_state_dict(checkpoint['optim_Denoise_net_state_dict'])
    optimizer_Decom.load_state_dict(checkpoint['optimizer_Decom_state_dict'])


    optimizer_L2H.load_state_dict(checkpoint['optimizer_L2H_state_dict'])
    optimizer_H2L.load_state_dict(checkpoint['optimizer_H2L_state_dict'])
    optimizer_D_L_low.load_state_dict(checkpoint['optimizer_D_L_low_state_dict'])
    optimizer_D_L_high.load_state_dict(checkpoint['optimizer_D_L_high_state_dict'])
    optimizer_D_R.load_state_dict(checkpoint['optimizer_D_R_state_dict'])

    # 获取训练信息
    start_epoch = checkpoint['epoch']
    loss_history = checkpoint.get('loss_history', {})
    metric_history = checkpoint.get('metric_history', {})
    best_metric_info = checkpoint.get('best_metric_info', {})

    print(f" 检查点已加载：{checkpoint_path}")
    print(f" 从 Epoch {start_epoch} 继续训练")

    return start_epoch, loss_history, metric_history, best_metric_info
