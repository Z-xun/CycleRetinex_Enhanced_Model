from PIL import Image
from model import *
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as transforms

device = "cpu"

Decom_net = Decomposition().to(device)
L2H_net = LCNet(mode="brighten").to(device)
Denoise_net = UNetDenoise().to(device)

checkpoint = torch.load("./best_model_v2.pth",map_location=torch.device('cpu'))
Decom_net.load_state_dict(checkpoint['Decom_net_state_dict'])
Denoise_net.load_state_dict(checkpoint['Denoise_net_state_dict'])
L2H_net.load_state_dict(checkpoint['L2H_net_state_dict'])

Decom_net.eval()
Denoise_net.eval()
L2H_net.eval()

transform =transforms.ToTensor()
img = transform(Image.open("./Night_data/low/9.png").convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    R,L = Decom_net(img)
    L_enh = L2H_net(L)
    L_enh = L_enh.expand(-1, 3, -1, -1) 
    R_denoise = Denoise_net(R)
    img_enh = L_enh * R_denoise

save_image(img_enh,"night_data_enh.png")