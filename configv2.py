import torch

generator_lr = 1e-4
discriminator_lr = 5e-5
lr = generator_lr
betas = (0.5, 0.999)
epochs = 100
batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

train_root = "./Night_data"
save_dir = "./working2"
sample_dir = "./samples2"
resume = False
resume_epoch = 0
resume_checkpoint_path = "./working2/checkpoint_epoch_30.pth"

lambda_cycle = 10.0
lambda_adv = 0.5
lambda_smooth_L = 1.0
lambda_recon = 10.0
lambda_temporal = 5.0
lambda_denoise_id = 1.0
lambda_illumination_prior = 3.0
lambda_illumination_stats = 1.5
lambda_illumination_order = 4.0
illumination_order_margin = 0.08

early_stop_patience = 20
