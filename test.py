import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from model import Decomposition, LCNet,UNetDenoise
from config import save_dir

CHECKPOINT_PATH = './best_model.pth'

BASE_DATA_DIR = './LOL-v2/Real_captured/Train'
LOW_IMAGE_NAME = 'low00006.png'   
HIGH_IMAGE_NAME = 'normal00019.png' 

OUTPUT_DIR = 'results/pair_comparison'
# ===============================================================

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片：{image_path}")
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    return img, transform(img).unsqueeze(0)

def tensor_to_rgb(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))

def tensor_to_gray(tensor):
    single_ch = tensor[:, 0:1, :, :].squeeze(0).squeeze(0).cpu().numpy()
    return Image.fromarray(np.clip(single_ch * 255, 0, 255).astype(np.uint8), mode='L')

def create_visualization(img_name, mode, img_pil, R, L,L_re, Rec, Enh, output_dir):

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    
    images = [img_pil, tensor_to_rgb(R), tensor_to_gray(L), tensor_to_rgb(L_re),tensor_to_rgb(Rec), tensor_to_rgb(Enh)]
    titles = [
        f"Input ({mode})",
        "Reflectance (R)",
        "Illumination (L)",
        "Re_L",
        "Reconstructed (R×L)",
        "Enhanced Result"
    ]
    
    for ax, img, title in zip(axes, images, titles):
        cmap = 'gray' if 'Illumination' in title else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_name = f"result_{mode}_{os.path.splitext(img_name)[0]}.png"
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.close(fig)
    
    return save_path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Decom_net = Decomposition().to(device)

    Denoise_R = UNetDenoise().to(device)
    L2H_net = LCNet(mode="brighten").to(device)
    H2L_net = LCNet(mode="darken").to(device)

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
        Decom_net.load_state_dict(checkpoint['Decom_net_state_dict'])
        Denoise_R.load_state_dict(checkpoint['Denoise_net_state_dict'])

        L2H_net.load_state_dict(checkpoint['L2H_net_state_dict'])
        H2L_net.load_state_dict(checkpoint['H2L_net_state_dict'])
        
        print(f"成功加载权重：{CHECKPOINT_PATH}")
    else:
        print(f"错误：找不到权重文件 {CHECKPOINT_PATH}")
        return

    Decom_net.eval()
    Denoise_R.eval()
    L2H_net.eval()
    H2L_net.eval()

    low_path = os.path.join(BASE_DATA_DIR, 'Low', LOW_IMAGE_NAME)

    high_path = os.path.join(BASE_DATA_DIR, 'Normal', HIGH_IMAGE_NAME)

    try:
        img_low_pil, img_low_t = load_image(low_path)
        img_high_pil, img_high_t = load_image(high_path)
    except FileNotFoundError as e:
        print(e)
        return

    img_low_t = img_low_t.to(device)
    img_high_t = img_high_t.to(device)

    print(f"\n 低光图：{LOW_IMAGE_NAME}")
    print(f" 高光图：{HIGH_IMAGE_NAME}")
    print("\n 开始推理...\n")

    with torch.no_grad():
        R_low, L_low = Decom_net(img_low_t)
        torchvision.utils.save_image(R_low,"R_low_raw.jpg")
        L_low_enhanced = L2H_net(L_low).expand(-1,3,-1,-1)
        L_low = L_low.expand(-1,3,-1,-1)
        R_low = Denoise_R(R_low)
        Rec_low = R_low * L_low
        Enh_low = R_low * L_low_enhanced

        R_high, L_high = Decom_net(img_high_t)
        L_high_reduced = H2L_net(L_high).expand(-1,3,-1,-1)
        L_high = L_high.expand(-1,3,-1,-1)
        Rec_high = R_high * L_high
        Enh_high = R_high * L_high_reduced

    print(" 生成可视化结果...\n")
    
    low_save_path = create_visualization(
        LOW_IMAGE_NAME, 'Low', img_low_pil,
        R_low, L_low, L_low_enhanced,Rec_low, Enh_low,
        OUTPUT_DIR
    )
    print(f" 低光图结果：{low_save_path}")
    torchvision.utils.save_image(L_low,"L_low.jpg")
    torchvision.utils.save_image(R_low,"R_low_denoise.jpg")
    img_max,_ = torch.max(img_low_t,dim = 1)
    torchvision.utils.save_image(img_max,"img_low_max.jpg")
    torchvision.utils.save_image(L_low_enhanced,"L_low_enhanced.jpg")

    torchvision.utils.save_image(L_high,"L_high.jpg")
    torchvision.utils.save_image(R_high,"R_high.jpg")
    torchvision.utils.save_image(L_high_reduced,"L_high_reduced.jpg")
    torchvision.utils.save_image(Enh_low,"low_enh.jpg")
    torchvision.utils.save_image(Enh_high,"high_enh.jpg")
    high_save_path = create_visualization(
        HIGH_IMAGE_NAME, 'Normal', img_high_pil,
        R_high, L_high,L_high_reduced, Rec_high, Enh_high,
        OUTPUT_DIR
    )
    print(f" 高光图结果：{high_save_path}")

    print(f"\n 完成！共生成 2 张独立结果图:")
    print(f"    {low_save_path}")
    print(f"    {high_save_path}\n")

    print(" 每张图包含 5 个子图:")
    print("   1. Input - 原始输入图像")
    print("   2. Reflectance (R) - 反射分量（材质/纹理）")
    print("   3. Illumination (L) - 光照分量（灰度图）")
    print("   4. Reconstructed - 重构图像 (R×L)")
    print("   5. Enhanced - 增强后图像\n")

if __name__ == "__main__":
    main()
