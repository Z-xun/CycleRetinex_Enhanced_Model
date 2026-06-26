import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import configv2 as cfg
from model import Decomposition, UNetDenoise, LCNet, Discriminator
from utils import (
    save_checkpoint,
    load_checkpoint,
    smooth_loss,
    illumination_prior_loss,
    illumination_statistics_loss,
    save_loss_plot,
    save_metric_plot,
)
from model import *
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VAL_RATIO = 0.1
MIN_VAL_SAMPLES = 8
SPLIT_SEED = 42
PAIRING_SEED = 1234


class FixedUnpairedDataset(Dataset):
    def __init__(self, low_paths, high_paths, pair_seed=PAIRING_SEED):
        super().__init__()
        if len(low_paths) == 0 or len(high_paths) == 0:
            raise ValueError("low_paths and high_paths must not be empty")

        self.low_paths = list(low_paths)
        self.high_paths = list(high_paths)
        self.transforms = A.Compose([ToTensorV2()])

        rng = random.Random(pair_seed)
        self.high_indices = [rng.randrange(len(self.high_paths)) for _ in self.low_paths]

    def __len__(self):
        return len(self.low_paths)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        tensor = self.transforms(image=np.array(img))["image"].float() / 255.0
        return tensor

    def __getitem__(self, index):
        low_img = self._load_image(self.low_paths[index])
        high_img = self._load_image(self.high_paths[self.high_indices[index]])
        return low_img, high_img


def resolve_split_dir(root, split_name):
    candidates = [
        os.path.join(root, split_name),
        os.path.join(root, split_name.lower()),
        os.path.join(root, split_name.upper()),
        os.path.join(root, split_name.capitalize()),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

def list_image_paths(directory):
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def split_train_val(paths, val_ratio=VAL_RATIO, seed=SPLIT_SEED):
    if len(paths) < 2:
        return list(paths), list(paths)

    indices = list(range(len(paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_count = max(1, min(len(paths) - 1, int(round(len(paths) * val_ratio))))
    if len(paths) >= MIN_VAL_SAMPLES:
        val_count = max(val_count, min(MIN_VAL_SAMPLES, len(paths) - 1))

    val_indices = set(indices[:val_count])
    train_paths = [paths[i] for i in range(len(paths)) if i not in val_indices]
    val_paths = [paths[i] for i in range(len(paths)) if i in val_indices]
    return train_paths, val_paths


def build_night_loaders():
    low_dir = resolve_split_dir(cfg.train_root, "low")
    high_dir = resolve_split_dir(train_root, "Normal")

    low_paths = list_image_paths(low_dir)
    high_paths = list_image_paths(high_dir)
    
    train_low_paths, val_low_paths = split_train_val(low_paths)
    train_high_paths, val_high_paths = split_train_val(high_paths)

    train_dataset = FixedUnpairedDataset(
        train_low_paths, train_high_paths, pair_seed=PAIRING_SEED
    )
    val_dataset = FixedUnpairedDataset(
        val_low_paths, val_high_paths, pair_seed=PAIRING_SEED + 1
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"train low images: {len(train_low_paths)}")
    print(f"train normal images: {len(train_high_paths)}")
    print(f"val low images: {len(val_low_paths)}")
    print(f"val normal images: {len(val_high_paths)}")

    return train_loader, val_loader


def forward_enhancement(img_low, Decom_net, L2H_net, Denoise_net_R):
    R_low, L_low = Decom_net(img_low)
    L_low_enhanced = L2H_net(L_low)
    R_low_denoise = Denoise_net_R(R_low)
    img_final = torch.clamp(
        R_low_denoise * L_low_enhanced.expand(-1, 3, -1, -1), 0.0, 1.0
    )
    return img_final, R_low, L_low, L_low_enhanced, R_low_denoise


def illumination_order_loss(L_low, L_low_enhanced, margin=None):
    if margin is None:
        margin = cfg.illumination_order_margin
    low_mean = L_low.mean(dim=(2, 3))
    enhanced_mean = L_low_enhanced.mean(dim=(2, 3))
    return torch.relu(low_mean + margin - enhanced_mean).mean()


@torch.no_grad()
def evaluate_model(data_loader, Decom_net, L2H_net, H2L_net, Denoise_net_R):
    Decom_net.eval()
    L2H_net.eval()
    Denoise_net_R.eval()

    criterion_recon = nn.L1Loss()
    total_metrics = {
        "val_recon": 0.0,
        "val_cycle": 0.0,
        "val_illum_prior": 0.0,
        "val_illum_stats": 0.0,
        "val_illum_order": 0.0,
        "val_denoise_id": 0.0,
        "score": 0.0,
    }
    sample_count = 0

    for img_low_eval, img_high_eval in data_loader:
        img_low_eval = img_low_eval.to(device)
        img_high_eval = img_high_eval.to(device)

        R_low_eval, L_low_eval = Decom_net(img_low_eval)
        R_high_eval, L_high_eval = Decom_net(img_high_eval)

        L_low_enhanced = L2H_net(L_low_eval)
        L_high_reduced = H2L_net(L_high_eval)
        R_low_denoise = Denoise_net_R(R_low_eval)
        R_high_denoise = Denoise_net_R(R_high_eval)

        L_low_cycle = H2L_net(L_low_enhanced)
        L_high_cycle = L2H_net(L_high_reduced)

        recon_low = criterion_recon(
            R_low_eval * L_low_eval.expand(-1, 3, -1, -1), img_low_eval
        )
        recon_high = criterion_recon(
            R_high_eval * L_high_eval.expand(-1, 3, -1, -1), img_high_eval
        )
        val_recon = recon_low + recon_high

        val_cycle = criterion_recon(L_low_cycle, L_low_eval) + criterion_recon(
            L_high_cycle, L_high_eval
        )

        val_illum_prior = illumination_prior_loss(
            L_low_eval, img_low_eval
        ) + illumination_prior_loss(L_high_eval, img_high_eval)

        val_illum_stats = illumination_statistics_loss(
            L_low_enhanced, L_high_eval
        ) + illumination_statistics_loss(L_high_reduced, L_low_eval)
        val_illum_order = illumination_order_loss(L_low_eval, L_low_enhanced)

        val_denoise_id = criterion_recon(R_high_denoise, R_high_eval)

        total_val_loss = (
            cfg.lambda_recon * val_recon
            + cfg.lambda_cycle * val_cycle
            + cfg.lambda_illumination_prior * val_illum_prior
            + cfg.lambda_illumination_stats * val_illum_stats
            + cfg.lambda_illumination_order * val_illum_order
            + cfg.lambda_denoise_id * val_denoise_id
        )

        batch_size_cur = img_low_eval.size(0)
        total_metrics["val_recon"] += val_recon.item() * batch_size_cur
        total_metrics["val_cycle"] += val_cycle.item() * batch_size_cur
        total_metrics["val_illum_prior"] += val_illum_prior.item() * batch_size_cur
        total_metrics["val_illum_stats"] += val_illum_stats.item() * batch_size_cur
        total_metrics["val_illum_order"] += val_illum_order.item() * batch_size_cur
        total_metrics["val_denoise_id"] += val_denoise_id.item() * batch_size_cur
        total_metrics["score"] += (-total_val_loss.item()) * batch_size_cur
        sample_count += batch_size_cur

    Decom_net.train()
    L2H_net.train()
    Denoise_net_R.train()

    if sample_count == 0:
        return {key: 0.0 for key in total_metrics}

    return {key: value / sample_count for key, value in total_metrics.items()}


def set_optimizer_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def save_visualization_grid(R_low, L_low_enhanced, output_path):
    img_final = torch.clamp(
        R_low * L_low_enhanced.expand(-1, 3, -1, -1), 0.0, 1.0
    )
    grid = torch.cat([R_low, L_low_enhanced.expand(-1, 3, -1, -1), img_final], dim=0)
    grid = make_grid(grid, nrow=3, padding=2, pad_value=1.0)
    save_image(grid, output_path)


def save_test_metrics(test_metrics, best_metric_info):
    result_path = os.path.join(cfg.save_dir, "final_test_metrics.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("note: validation uses an unpaired split of Night_data.\n")
        f.write(f"best_epoch: {best_metric_info['epoch']}\n")
        f.write(f"best_val_recon: {best_metric_info['val_recon']:.6f}\n")
        f.write(f"best_val_cycle: {best_metric_info['val_cycle']:.6f}\n")
        f.write(f"best_val_illum_prior: {best_metric_info['val_illum_prior']:.6f}\n")
        f.write(f"best_val_illum_stats: {best_metric_info['val_illum_stats']:.6f}\n")
        f.write(f"best_val_illum_order: {best_metric_info['val_illum_order']:.6f}\n")
        f.write(f"best_val_denoise_id: {best_metric_info['val_denoise_id']:.6f}\n")
        f.write(f"best_val_score: {best_metric_info['score']:.6f}\n")
        f.write(f"val_recon: {test_metrics['val_recon']:.6f}\n")
        f.write(f"val_cycle: {test_metrics['val_cycle']:.6f}\n")
        f.write(f"val_illum_prior: {test_metrics['val_illum_prior']:.6f}\n")
        f.write(f"val_illum_stats: {test_metrics['val_illum_stats']:.6f}\n")
        f.write(f"val_illum_order: {test_metrics['val_illum_order']:.6f}\n")
        f.write(f"val_denoise_id: {test_metrics['val_denoise_id']:.6f}\n")
        f.write(f"val_score: {test_metrics['score']:.6f}\n")
    print(f"Final validation metrics saved to {result_path}")


def train():
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("./vis", exist_ok=True)
    os.makedirs("./vis2", exist_ok=True)

    Decom_net = Decomposition().to(device)
    Denoise_net_R = UNetDenoise(in_channels=3, out_channels=3).to(device)
    L2H_net = LCNet(mode="brighten").to(device)
    H2L_net = LCNet(mode="darken").to(device)

    D_L_low = Discriminator(input_channels=1).to(device)
    D_L_high = Discriminator(input_channels=1).to(device)
    D_R = Discriminator(input_channels=3).to(device)

    optimizer_Decom = optim.Adam(Decom_net.parameters(), lr=generator_lr, betas=betas)
    optimizer_L2H = optim.Adam(L2H_net.parameters(), lr=generator_lr, betas=betas)
    optimizer_H2L = optim.Adam(H2L_net.parameters(), lr=generator_lr, betas=betas)
    optimizer_Denoise_R = optim.Adam(
        Denoise_net_R.parameters(), lr=generator_lr, betas=betas
    )

    optimizer_D_L_low = optim.Adam(
        D_L_low.parameters(), lr=discriminator_lr, betas=betas
    )
    optimizer_D_L_high = optim.Adam(
        D_L_high.parameters(), lr=discriminator_lr, betas=betas
    )
    optimizer_D_R = optim.Adam(D_R.parameters(), lr=discriminator_lr, betas=betas)

    train_loader_unpaired, val_loader = build_night_loaders()

    criterion_GAN = nn.MSELoss()
    criterion_recon = nn.L1Loss()

    epoch_loss_template = {
        "G_total": [],
        "D_total": [],
        "cycle": [],
        "G_L": [],
        "G_R": [],
        "recon": [],
        "illum_prior": [],
        "illum_stats": [],
        "illum_order": [],
        "D_L_low": [],
        "D_L_high": [],
        "D_R": [],
    }
    epoch_loss_history = {key: [] for key in epoch_loss_template}
    metric_history = {
        "val_recon": [],
        "val_cycle": [],
        "val_illum_prior": [],
        "val_illum_stats": [],
        "val_illum_order": [],
        "val_denoise_id": [],
        "score": [],
    }
    best_metric_info = {
        "epoch": -1,
        "val_recon": 0.0,
        "val_cycle": 0.0,
        "val_illum_prior": 0.0,
        "val_illum_stats": 0.0,
        "val_illum_order": 0.0,
        "val_denoise_id": 0.0,
        "score": float("-inf"),
    }

    start_epoch = 0
    checkpoint_path = cfg.resume_checkpoint_path.strip()
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_result = load_checkpoint(
            checkpoint_path,
            Decom_net=Decom_net,
            L2H_net=L2H_net,
            H2L_net=H2L_net,
            Denoise_net=Denoise_net_R,
            D_L_low=D_L_low,
            D_L_high=D_L_high,
            D_R=D_R,
            optimizer_Denoise_net=optimizer_Denoise_R,
            optimizer_Decom=optimizer_Decom,
            optimizer_L2H=optimizer_L2H,
            optimizer_H2L=optimizer_H2L,
            optimizer_D_L_low=optimizer_D_L_low,
            optimizer_D_L_high=optimizer_D_L_high,
            optimizer_D_R=optimizer_D_R,
            device=device,
        )
        start_epoch, loaded_loss_history, loaded_metric_history, loaded_best_metric_info = (
            checkpoint_result
        )
        for key in epoch_loss_template.keys():
            epoch_loss_history[key] = loaded_loss_history.get(key, [])
        for key in metric_history.keys():
            metric_history[key] = loaded_metric_history.get(key, [])
        best_metric_info.update(loaded_best_metric_info)

        for optimizer in (
            optimizer_Decom,
            optimizer_L2H,
            optimizer_H2L,
            optimizer_Denoise_R,
        ):
            set_optimizer_lr(optimizer, generator_lr)
        for optimizer in (optimizer_D_L_low, optimizer_D_L_high, optimizer_D_R):
            set_optimizer_lr(optimizer, discriminator_lr)

    last_R_low = None
    last_L_low_enhanced = None
    best_visual_saved_this_run = False
    print(start_epoch)
    for epoch in range(start_epoch, epochs):
        epoch_losses = {k: 0.0 for k in epoch_loss_history.keys()}

        for img_low_real, img_high_real in tqdm(train_loader_unpaired):
            img_low_real = img_low_real.to(device)
            img_high_real = img_high_real.to(device)

            R_low_real, L_low_real = Decom_net(img_low_real)
            R_high_real, L_high_real = Decom_net(img_high_real)

            L_low_real_enhanced = L2H_net(L_low_real)
            L_high_real_reduced = H2L_net(L_high_real)
            R_low_real_denoise = Denoise_net_R(R_low_real)

            L_low_cycle = H2L_net(L_low_real_enhanced)
            cycle_loss_low = criterion_recon(L_low_cycle, L_low_real.detach())

            L_high_cycle = L2H_net(L_high_real_reduced)
            cycle_loss_high = criterion_recon(L_high_cycle, L_high_real.detach())

            cycle_loss = (cycle_loss_low + cycle_loss_high) * lambda_cycle

            pred_L_high_fake = D_L_high(L_low_real_enhanced)
            G_L_high_loss = criterion_GAN(
                D_L_high(L_low_real_enhanced),
                torch.full_like(pred_L_high_fake, 0.9),
            )

            pred_L_low_fake = D_L_low(L_high_real_reduced)
            G_L_low_loss = criterion_GAN(
                D_L_low(L_high_real_reduced),
                torch.full_like(pred_L_low_fake, 0.9),
            )
            G_L_loss = G_L_low_loss + G_L_high_loss

            pred_R_low_id = D_R(R_low_real_denoise)
            G_R_loss = criterion_GAN(
                D_R(R_low_real_denoise),
                torch.full_like(pred_R_low_id, 0.9),
            )

            recon_loss_low = criterion_recon(
                R_low_real * L_low_real.expand(-1, 3, -1, -1), img_low_real
            )
            recon_loss_high = criterion_recon(
                R_high_real * L_high_real.expand(-1, 3, -1, -1), img_high_real
            )
            recon_loss_cycle = criterion_recon(
                L_high_cycle.expand(-1, 3, -1, -1) * R_high_real,
                img_high_real.detach(),
            ) + criterion_recon(
                L_low_cycle.expand(-1, 3, -1, -1) * R_low_real_denoise,
                img_low_real.detach(),
            )
            recon_loss = (
                recon_loss_low + recon_loss_high + recon_loss_cycle
            ) * lambda_recon
            loss_smooth = smooth_loss(L_low_real, img_low_real) + smooth_loss(
                L_high_real, img_high_real
            )
            loss_illumination_prior = (
                illumination_prior_loss(L_low_real, img_low_real)
                + illumination_prior_loss(L_high_real, img_high_real)
            )
            loss_illumination_stats = (
                illumination_statistics_loss(L_low_real_enhanced, L_high_real.detach())
                + illumination_statistics_loss(L_high_real_reduced, L_low_real.detach())
            ) * cfg.lambda_illumination_stats
            loss_illumination_order = (
                illumination_order_loss(L_low_real, L_low_real_enhanced)
                * cfg.lambda_illumination_order
            )

            R_high_real_id = Denoise_net_R(R_high_real)
            loss_denoise_id = criterion_recon(R_high_real_id, R_high_real.detach())

            G_total_loss = (
                cycle_loss
                + lambda_adv * (G_L_loss + G_R_loss)
                + recon_loss
                + lambda_smooth_L * loss_smooth
                + lambda_illumination_prior * loss_illumination_prior
                + loss_illumination_stats
                + loss_illumination_order
                + lambda_denoise_id * loss_denoise_id
            )

            optimizer_Decom.zero_grad()
            optimizer_L2H.zero_grad()
            optimizer_H2L.zero_grad()
            optimizer_Denoise_R.zero_grad()

            G_total_loss.backward()
            optimizer_Decom.step()
            optimizer_L2H.step()
            optimizer_H2L.step()
            optimizer_Denoise_R.step()

            d_loss_L_low = (
                criterion_GAN(
                    D_L_low(L_low_real.detach()),
                    torch.full_like(D_L_low(L_low_real.detach()), 0.9),
                )
                + criterion_GAN(
                    D_L_low(L_high_real_reduced.detach()),
                    torch.full_like(D_L_low(L_high_real_reduced.detach()), 0.1),
                )
            ) * 0.5

            optimizer_D_L_low.zero_grad()
            d_loss_L_low.backward()
            optimizer_D_L_low.step()

            d_loss_L_high = (
                criterion_GAN(
                    D_L_high(L_high_real.detach()),
                    torch.full_like(D_L_high(L_high_real.detach()), 0.9),
                )
                + criterion_GAN(
                    D_L_high(L_low_real_enhanced.detach()),
                    torch.full_like(D_L_high(L_low_real_enhanced.detach()), 0.1),
                )
            ) * 0.5

            optimizer_D_L_high.zero_grad()
            d_loss_L_high.backward()
            optimizer_D_L_high.step()

            d_loss_R = (
                criterion_GAN(
                    D_R(R_high_real.detach()),
                    torch.full_like(D_R(R_high_real.detach()), 0.9),
                )
                + criterion_GAN(
                    D_R(R_low_real_denoise.detach()),
                    torch.full_like(D_R(R_low_real_denoise.detach()), 0.1),
                )
            ) * 0.5

            optimizer_D_R.zero_grad()
            d_loss_R.backward()
            optimizer_D_R.step()

            epoch_losses["G_total"] += G_total_loss.item()
            epoch_losses["cycle"] += cycle_loss.item()
            epoch_losses["G_L"] += G_L_loss.item()
            epoch_losses["G_R"] += G_R_loss.item()
            epoch_losses["recon"] += recon_loss.item()
            epoch_losses["illum_prior"] += loss_illumination_prior.item()
            epoch_losses["illum_stats"] += loss_illumination_stats.item()
            epoch_losses["illum_order"] += loss_illumination_order.item()
            epoch_losses["D_L_low"] += d_loss_L_low.item()
            epoch_losses["D_L_high"] += d_loss_L_high.item()
            epoch_losses["D_R"] += d_loss_R.item()
            epoch_losses["D_total"] += (
                d_loss_L_low.item() + d_loss_L_high.item() + d_loss_R.item()
            )

            last_R_low = R_low_real_denoise.detach()
            last_L_low_enhanced = L_low_real_enhanced.detach()

        for key in epoch_loss_history.keys():
            epoch_loss_history[key].append(epoch_losses[key] / len(train_loader_unpaired))

        eval_metrics = evaluate_model(
            val_loader, Decom_net, L2H_net, H2L_net, Denoise_net_R
        )
        for key in metric_history.keys():
            metric_history[key].append(eval_metrics[key])

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"G_Loss: {epoch_loss_history['G_total'][-1]:.4f}, "
            f"D_Loss: {epoch_loss_history['D_total'][-1]:.4f}, "
            f"Val_Recon: {eval_metrics['val_recon']:.4f}, "
            f"Val_Cycle: {eval_metrics['val_cycle']:.4f}, "
            f"Val_Score: {eval_metrics['score']:.4f}"
        )

        if eval_metrics["score"] > best_metric_info["score"]:
            best_metric_info = {
                "epoch": epoch + 1,
                "val_recon": eval_metrics["val_recon"],
                "val_cycle": eval_metrics["val_cycle"],
                "val_illum_prior": eval_metrics["val_illum_prior"],
                "val_illum_stats": eval_metrics["val_illum_stats"],
                "val_illum_order": eval_metrics["val_illum_order"],
                "val_denoise_id": eval_metrics["val_denoise_id"],
                "score": eval_metrics["score"],
            }
            best_model_path = os.path.join(cfg.save_dir, "best_model_v2.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "Decom_net_state_dict": Decom_net.state_dict(),
                    "L2H_net_state_dict": L2H_net.state_dict(),
                    "H2L_net_state_dict": H2L_net.state_dict(),
                    "Denoise_net_state_dict": Denoise_net_R.state_dict(),
                    "loss_history": epoch_loss_history,
                    "metric_history": metric_history,
                    "best_metric_info": best_metric_info,
                },
                best_model_path,
            )
            if last_R_low is not None and last_L_low_enhanced is not None:
                save_visualization_grid(
                    last_R_low,
                    last_L_low_enhanced,
                    f"./vis2/best_visual_epoch_{epoch + 1}_v2.png",
                )
                save_visualization_grid(
                    last_R_low,
                    last_L_low_enhanced,
                    "./vis2/best_visual_latest_v2.png",
                )
                best_visual_saved_this_run = True
            print(
                f"Best model updated at epoch {epoch + 1}: "
                f"Val_Recon={eval_metrics['val_recon']:.4f}, "
                f"Val_Cycle={eval_metrics['val_cycle']:.4f}, "
                f"Val_Score={eval_metrics['score']:.4f}"
            )

        save_checkpoint_c = False
        if (epoch + 1) <= epochs * 0.5 and (epoch + 1) % int(0.1 * epochs) == 0:
            save_checkpoint_c = True
        elif (
            epochs * 0.5 < (epoch + 1) <= epochs * 0.9
            and (epoch + 1) % int(0.05 * epochs) == 0
        ):
            save_checkpoint_c = True
        elif 0.9 * epochs < (epoch + 1) <= epochs:
            save_checkpoint_c = True

        if save_checkpoint_c:
            save_checkpoint(
                epoch + 1,
                Decom_net,
                L2H_net,
                H2L_net,
                Denoise_net_R,
                D_L_low,
                D_L_high,
                D_R,
                optimizer_Decom,
                optimizer_L2H,
                optimizer_H2L,
                optimizer_Denoise_R,
                optimizer_D_L_low,
                optimizer_D_L_high,
                optimizer_D_R,
                epoch_loss_history,
                metric_history,
                best_metric_info,
                cfg.save_dir,
            )
            print(f"checkpoint_epoch_{epoch + 1}.pth saved")

        if (epoch + 1) % 5 == 0 and last_R_low is not None:
            save_visualization_grid(
                last_R_low,
                last_L_low_enhanced,
                f"./vis2/compare_epoch_{epoch + 1}_v2.png",
            )

    if not best_visual_saved_this_run:
        print("No new best visualization epoch in this run; keeping periodic visualization snapshots.")

    save_loss_plot(epoch_loss_history, cfg.save_dir)
    save_metric_plot(metric_history, cfg.save_dir)

    best_model_path = os.path.join(cfg.save_dir, "best_model_v2.pth")
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        Decom_net.load_state_dict(best_checkpoint["Decom_net_state_dict"])
        L2H_net.load_state_dict(best_checkpoint["L2H_net_state_dict"])
        H2L_net.load_state_dict(best_checkpoint["H2L_net_state_dict"])
        Denoise_net_R.load_state_dict(best_checkpoint["Denoise_net_state_dict"])
        val_metrics = evaluate_model(
            val_loader, Decom_net, L2H_net, H2L_net, Denoise_net_R
        )
        save_test_metrics(val_metrics, best_checkpoint["best_metric_info"])
        print(
            f"Training finished. Best epoch: {best_checkpoint['best_metric_info']['epoch']}, "
            f"Best_Val_Recon: {best_checkpoint['best_metric_info']['val_recon']:.4f}, "
            f"Best_Val_Cycle: {best_checkpoint['best_metric_info']['val_cycle']:.4f}, "
            f"Val_Recon: {val_metrics['val_recon']:.4f}, "
            f"Val_Cycle: {val_metrics['val_cycle']:.4f}"
        )


if __name__ == "__main__":
    train()
