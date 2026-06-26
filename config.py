import torch

generator_lr = 1e-4
discriminator_lr = 5e-5
lr = generator_lr
betas = (0.5, 0.999)
epochs = 100
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

train_root = "./LOL-v2/Real_captured/Train"
test_root = "./LOL-v2/Real_captured/Test"
save_dir = "./working"
sample_dir = "./samples"
resume = False
resume_epoch = 0
resume_checkpoint_path = ""

lambda_cycle = 10.0
lambda_adv = 0.05
lambda_smooth_L = 2.0
lambda_recon = 10.0
lambda_denoise_id = 1.0
lambda_reflectance = 1.0
lambda_illumination_order = 1.0
lambda_illumination_prior = 3.0
lambda_illumination_match = 5.0
lambda_cross_recon = 5.0


