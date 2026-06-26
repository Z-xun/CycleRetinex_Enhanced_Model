import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from model import *
from config import *
from utils import *


def forward_enhancement(img_low, Decom_net, L2H_net, Denoise_net_R):
    R_low, L_low = Decom_net(img_low)
    L_low_enhanced = L2H_net(L_low)
    R_low_denoise = Denoise_net_R(R_low)
    img_final = torch.clamp(
        R_low_denoise * L_low_enhanced.expand(-1, 3, -1, -1), 0.0, 1.0
    )
    return img_final, R_low, L_low, L_low_enhanced, R_low_denoise


@torch.no_grad()
def evaluate_model(data_loader, Decom_net, L2H_net, Denoise_net_R):
    Decom_net.eval()
    L2H_net.eval()
    Denoise_net_R.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    sample_count = 0

    for img_low_eval, img_high_eval in data_loader:
        img_low_eval = img_low_eval.to(device)
        img_high_eval = img_high_eval.to(device)

        img_enhanced, _, _, _, _ = forward_enhancement(
            img_low_eval, Decom_net, L2H_net, Denoise_net_R
        )

        batch_size_cur = img_low_eval.size(0)
        total_psnr += calculate_psnr(img_enhanced, img_high_eval) * batch_size_cur
        total_ssim += calculate_ssim(img_enhanced, img_high_eval) * batch_size_cur
        sample_count += batch_size_cur

    Decom_net.train()
    L2H_net.train()
    Denoise_net_R.train()

    if sample_count == 0:
        return {"psnr": 0.0, "ssim": 0.0, "score": 0.0}

    avg_psnr = total_psnr / sample_count
    avg_ssim = total_ssim / sample_count

    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "score": avg_psnr + 10.0 * avg_ssim,
    }


def build_paired_loader(split_root, shuffle=False):
    dataset = MyDataset_paired(split_root + "/Low", split_root + "/Normal")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def illumination_order_loss(L_low, L_high, margin=0.05):
    return torch.relu(L_low - L_high + margin).mean()


def set_optimizer_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def save_test_metrics(test_metrics, best_metric_info):
    result_path = os.path.join(save_dir, "final_test_metrics.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"best_epoch: {best_metric_info['epoch']}\n")
        f.write(f"best_test_psnr: {best_metric_info['psnr']:.6f}\n")
        f.write(f"best_test_ssim: {best_metric_info['ssim']:.6f}\n")
        f.write(f"best_test_score: {best_metric_info['score']:.6f}\n")
        f.write(f"test_psnr: {test_metrics['psnr']:.6f}\n")
        f.write(f"test_ssim: {test_metrics['ssim']:.6f}\n")
        f.write(f"test_score: {test_metrics['score']:.6f}\n")
    print(f"Final test metrics saved to {result_path}")


def train():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./vis", exist_ok=True)

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

    train_loader = build_paired_loader(train_root, shuffle=True)

    print(f"Evaluation split: {test_root}")
    test_loader = build_paired_loader(test_root, shuffle=False)

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
        "illum_match": [],
        "cross_recon": [],
        "D_L_low": [],
        "D_L_high": [],
        "D_R": [],
    }
    epoch_loss_history = {key: [] for key in epoch_loss_template}
    metric_history = {
        "psnr": [],
        "ssim": [],
        "score": [],
    }
    best_metric_info = {
        "epoch": -1,
        "psnr": 0.0,
        "ssim": 0.0,
        "score": float("-inf"),
    }

    start_epoch = 0
    checkpoint_path = resume_checkpoint_path.strip()
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
    last_L_low = None
    last_L_low_enhanced = None

    for epoch in range(start_epoch, epochs):
        epoch_losses = {k: 0.0 for k in epoch_loss_history.keys()}
        curloader = train_loader

        for img_low_real, img_high_real in tqdm(curloader):
            img_low_real = img_low_real.to(device)
            img_high_real = img_high_real.to(device)

            R_low_real, L_low_real = Decom_net(img_low_real)
            R_high_real, L_high_real = Decom_net(img_high_real)

            L_low_real_enhanced = L2H_net(L_low_real)
            L_high_real_reduced = H2L_net(L_high_real)
            R_low_real_denoise = Denoise_net_R(R_low_real)

            illum_target_low = illumination_prior_map(img_low_real).detach()
            illum_target_high = illumination_prior_map(img_high_real).detach()

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
            loss_illumination_match = (
                criterion_recon(L_low_real_enhanced, illum_target_high)
                + criterion_recon(L_high_real_reduced, illum_target_low)
            ) * lambda_illumination_match
            loss_cross_recon = (
                criterion_recon(
                    R_low_real_denoise * L_high_real.expand(-1, 3, -1, -1),
                    img_high_real,
                )
                + criterion_recon(
                    R_high_real * L_low_real.expand(-1, 3, -1, -1),
                    img_low_real,
                )
            ) * lambda_cross_recon
            loss_reflectance = criterion_recon(
                R_low_real_denoise, R_high_real.detach()
            ) * lambda_reflectance
            loss_illumination_order = illumination_order_loss(
                L_low_real, L_high_real
            ) * lambda_illumination_order

            R_high_real_id = Denoise_net_R(R_high_real)
            loss_denoise_id = criterion_recon(R_high_real_id, R_high_real.detach())

            G_total_loss = (
                cycle_loss
                + lambda_adv * (G_L_loss + G_R_loss)
                + recon_loss
                + lambda_smooth_L * loss_smooth
                + lambda_illumination_prior * loss_illumination_prior
                + loss_illumination_match
                + loss_cross_recon
                + loss_reflectance
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
            epoch_losses["illum_match"] += loss_illumination_match.item()
            epoch_losses["cross_recon"] += loss_cross_recon.item()
            epoch_losses["D_L_low"] += d_loss_L_low.item()
            epoch_losses["D_L_high"] += d_loss_L_high.item()
            epoch_losses["D_R"] += d_loss_R.item()
            epoch_losses["D_total"] += (
                d_loss_L_low.item() + d_loss_L_high.item() + d_loss_R.item()
            )

            last_R_low = R_low_real_denoise.detach()
            last_L_low = L_low_real.expand(-1, 3, -1, -1).detach()
            last_L_low_enhanced = L_low_real_enhanced.detach()

        for key in epoch_loss_history.keys():
            epoch_loss_history[key].append(epoch_losses[key] / len(curloader))

        eval_metrics = evaluate_model(test_loader, Decom_net, L2H_net, Denoise_net_R)
        for key in metric_history.keys():
            metric_history[key].append(eval_metrics[key])

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"G_Loss: {epoch_loss_history['G_total'][-1]:.4f}, "
            f"D_Loss: {epoch_loss_history['D_total'][-1]:.4f}, "
            f"Test_PSNR: {eval_metrics['psnr']:.4f}, "
            f"Test_SSIM: {eval_metrics['ssim']:.4f}, "
            f"Test_Score: {eval_metrics['score']:.4f}"
        )

        if eval_metrics["score"] > best_metric_info["score"]:
            best_metric_info = {
                "epoch": epoch + 1,
                "psnr": eval_metrics["psnr"],
                "ssim": eval_metrics["ssim"],
                "score": eval_metrics["score"],
            }
            best_model_path = os.path.join(save_dir, "best_model.pth")
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
            print(
                f"Best model updated at epoch {epoch + 1}: "
                f"Test_PSNR={eval_metrics['psnr']:.4f}, Test_SSIM={eval_metrics['ssim']:.4f}"
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
                save_dir,
            )
            print(f"checkpoint_epoch_{epoch + 1}.pth saved")

        if (epoch + 1) % 5 == 0 and last_R_low is not None:
            img_final = torch.clamp(
                last_R_low * last_L_low_enhanced.expand(-1, 3, -1, -1), 0.0, 1.0
            )
            grid = torch.cat(
                [
                    last_R_low,
                    last_L_low,
                    img_final,
                ],
                dim=0,
            )
            grid = make_grid(grid, nrow=3, padding=2, pad_value=1.0)
            save_image(grid, f"./vis/compare_epoch_{epoch + 1}.png")

    save_loss_plot(epoch_loss_history, save_dir)
    save_metric_plot(metric_history, save_dir)

    best_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        Decom_net.load_state_dict(best_checkpoint["Decom_net_state_dict"])
        L2H_net.load_state_dict(best_checkpoint["L2H_net_state_dict"])
        H2L_net.load_state_dict(best_checkpoint["H2L_net_state_dict"])
        Denoise_net_R.load_state_dict(best_checkpoint["Denoise_net_state_dict"])
        test_metrics = evaluate_model(test_loader, Decom_net, L2H_net, Denoise_net_R)
        save_test_metrics(test_metrics, best_checkpoint["best_metric_info"])
        print(
            f"Training finished. Best epoch: {best_checkpoint['best_metric_info']['epoch']}, "
            f"Best_Test_PSNR: {best_checkpoint['best_metric_info']['psnr']:.4f}, "
            f"Best_Test_SSIM: {best_checkpoint['best_metric_info']['ssim']:.4f}, "
            f"Test_PSNR: {test_metrics['psnr']:.4f}, "
            f"Test_SSIM: {test_metrics['ssim']:.4f}"
        )


if __name__ == "__main__":
    train()
