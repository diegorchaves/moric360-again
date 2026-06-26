import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# nohup python -u train.py --context_arm 24 --dim_arm_mod 24 --lambda_rate_list 1e-3 --sythesis_features 24 --type kodak > test.out&

import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from lossy_contour_algorithm import get_border_bits
from models.candidate_train import train_with_candidates
from models.model import Masked_INR
from utils.eval_model import (
    compute_ws_mse,
    compute_ws_psnr,
    compute_ws_ssim,
    eval_model,
)

manual_seed = 1


def seed_everything(seed=1029):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PATHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(1)

print("seed", manual_seed)


def get_mgrid(w_sidelen, h_sidelen, dim=2):

    x = torch.linspace(-1, 1, steps=w_sidelen)
    y = torch.linspace(-1, 1, steps=h_sidelen)
    tensors = (x, y) if dim == 2 else (x,) * dim

    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)

    mgrid = mgrid.unsqueeze(0).permute(0, 3, 2, 1)

    return mgrid


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")
    return 0


def loss_to_psnr(loss, max=1):
    return 10 * np.log10(max**2 / np.asarray(loss))


def get_mask_h_w(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    y_indices, x_indices = np.where(mask == 255)
    target_mask = mask == 255

    if len(x_indices) > 0 and len(y_indices) > 0:
        min_x, max_x = x_indices.min(), x_indices.max()
        min_y, max_y = y_indices.min(), y_indices.max()

        width = max_x - min_x + 1
        height = max_y - min_y + 1
        cropped_mask = target_mask[min_y : max_y + 1, min_x : max_x + 1]
    return width, height, torch.from_numpy(cropped_mask).unsqueeze(0).unsqueeze(0)


def mm(mask_path, h_img, w_img):
    if args.mask_type == "erp":
        # Polar regions (top 25% and bottom 25%) are foreground (True)
        # Equatorial region (middle 50%) is background (False)
        target_mask_4d = torch.ones(1, 1, h_img, w_img, dtype=torch.bool)
        top = int(h_img * 0.25)
        bottom = int(h_img * 0.75)
        target_mask_4d[:, :, top:bottom, :] = False
        target_mask_flat = target_mask_4d.view(-1)

    elif args.mask_type == "full":
        target_mask_4d = torch.ones((1, 1, h_img, w_img), dtype=torch.bool)
        target_mask_flat = target_mask_4d.view(-1)

    elif args.mask_type == "original":
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        target_mask = mask == 255

        target_mask_flat = torch.from_numpy(target_mask.flatten()).bool()
        target_mask_4d = torch.from_numpy(target_mask).unsqueeze(0).unsqueeze(0).bool()

    return target_mask_flat, target_mask_4d


def train(
    target_mask,
    model,
    dataloader,
    total_steps,
    total_steps_2,
    steps_til_summary,
    img_index,
    saved_path,
):

    vis_colum = 3
    best_psnr = 0
    if args.wsmse_tag == 1:
        criterion = compute_ws_mse
    elif args.wsmse_tag == 0:
        criterion = nn.MSELoss().cuda()
    base_params = [p for name, p in model.named_parameters()]

    optim = torch.optim.Adam([{"params": base_params, "lr": args.lr}])
    scheduler = CosineAnnealingLR(optim, T_max=total_steps)

    optimizer_stage_2 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    scheduler_stage_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stage_2, mode="min", factor=0.8, patience=20
    )

    for batch_idx, (img_in, _) in enumerate(dataloader, 0):
        model.train()
        vis_colum = 1
        batch_size, _, height, width = img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size, -1, 3).cuda()
        pixels1 = pixels[:, target_mask, :]
        pixels2 = pixels[:, ~target_mask, :]

        coords = get_mgrid(width // args.scale, height // args.scale, 2).cuda()
        losses = []
        losses_2 = []

        initial_noise_param = 2.0
        final_noise_param = 1.0
        initial_temperature = 0.3
        final_temperature = 0.1
        print(
            "start temperature:",
            initial_temperature,
            "noise parameter:",
            initial_noise_param,
        )
        print(
            "end temperature:", final_temperature, "noise parameter:", final_noise_param
        )

        print("********************Start from stage I")

        best_rd = 1000
        for step in range(total_steps + 1):
            ###stage 1:
            model.train()

            model.noise_parameter = initial_noise_param - (step / total_steps) * (
                initial_noise_param - final_noise_param
            )
            model.soft_round_temperature = initial_temperature - (
                step / total_steps
            ) * (initial_temperature - final_temperature)

            model_output, rate, _ = model(coords)

            bits_rate = rate.sum() / (args.all_pix_num)

            out_full = model_output.squeeze(0).view(height, width, 3)
            target_full = pixels.squeeze(0).view(height, width, 3)
            loss_mse = criterion(out_full, target_full)

            loss = args.lambda_rate * bits_rate + loss_mse
            losses.append(loss.item())
            if not step % steps_til_summary or (step == total_steps - 1):
                psnr_this_iter = compute_ws_psnr(out_full, target_full)

                if (loss < best_rd) and (step > 0):
                    best_psnr = psnr_this_iter
                    best_rd = loss
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "binary mask": None,
                    }

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                10,
                norm_type=2.0,
                error_if_nonfinite=False,
            )

            optim.step()
            scheduler.step()
        torch.save(checkpoint, saved_path)
        checkpoints = torch.load(saved_path)
        model.load_state_dict(checkpoints["model_state_dict"])
        ###stage_2
        print("********************going into stage II")
        best_psnr_2 = 0
        best_rd_2 = 1000
        for step in range(total_steps_2):
            model.train()
            model.quantizer_type = "softround_alone"
            model.quantizer_noise_type = "none"
            model.soft_round_temperature = 1e-4

            model_output, rate, _ = model(coords)

            bits_rate = rate.sum() / (args.all_pix_num)
            out_full = model_output.squeeze(0).view(height, width, 3)
            target_full = pixels.squeeze(0).view(height, width, 3)
            loss_mse = criterion(out_full, target_full)

            loss_2 = args.lambda_rate * bits_rate + loss_mse
            losses_2.append(loss_2.item())
            if not step % steps_til_summary or (step == total_steps_2 - 1):
                psnr_this_iter = compute_ws_psnr(out_full, target_full)
                if (loss_2 < best_rd_2) and (step > 0):
                    best_psnr_2 = psnr_this_iter
                    best_rd_2 = loss_2
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "binary mask": None,
                    }

            optimizer_stage_2.zero_grad()
            loss_2.backward()
            optimizer_stage_2.step()
            scheduler_stage_2.step(loss_2)
            current_lr = optimizer_stage_2.param_groups[0]["lr"]

            if current_lr < 1e-8:
                print(f"Current learning rate: {current_lr}")
                print(
                    f"Stopping training early: Learning rate has dropped below lr_threshold"
                )
                break

        torch.cuda.empty_cache()

        model.eval()

        model_output, rate, binary_mask = model(coords)

        bits_rate_eval = rate.sum() / (args.all_pix_num)
        bits_rate_eval_num = rate.sum()

        out_full = model_output.squeeze(0).view(height, width, 3)
        target_full = pixels.squeeze(0).view(height, width, 3)
        loss_mse = criterion(out_full, target_full)

        # 1. Transformar a máscara achatada de volta para 2D (Altura, Largura)
        mask_2d = target_mask.view(height, width)

        # 2. Copia as imagens e força o fundo a ser idêntico (erro = 0 no fundo)
        out_obj = out_full.clone()
        target_obj = target_full.clone()
        out_obj[~mask_2d] = target_obj[~mask_2d]

        eval_o = compute_ws_psnr(out_obj, target_obj)
        print("eval_object_psnr:", eval_o)

        # Copia as imagens e força o objeto a ser idêntico (erro = 0 no objeto)
        out_bg = out_full.clone()
        target_bg = target_full.clone()
        out_bg[mask_2d] = target_bg[mask_2d]

        loss_mse_b = criterion(out_bg, target_bg)
        eval_b = compute_ws_psnr(out_bg, target_bg)
        print("eval_background_psnr:", eval_b)

        print("eval_background_psnr:", eval_b)
        psnr_eval = compute_ws_psnr(out_full, target_full)
        print(
            "********************Evaluation the Image %d-th, after Step %d, BEST PSNR: %0.6f, Print rate %0.6f. *************************"
            % (img_index, step, psnr_eval, bits_rate_eval.item())
        )

        torch.cuda.empty_cache()

        torch.save(checkpoint, saved_path)
        print("Saved model at", saved_path)
    return psnr_eval, bits_rate_eval.item(), bits_rate_eval_num.item()


global args
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Example")
parser.add_argument("--batch_size", type=int, default=1, help="Batch-size")
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="learning rate"
)
parser.add_argument(
    "--data", type=str, default="../data", help="Location to store data"
)
parser.add_argument("--sparsity", type=float, default=0.0, help="prune rate")
parser.add_argument(
    "--local_upsampling_kernel_size", type=int, default=8, help="2, 4 or 8"
)
parser.add_argument("--upsampling_kernel_size", type=int, default=8, help="2, 4 or 8")
parser.add_argument(
    "--static_upsampling_kernel",
    default=False,
    help="Use this flag to **not** learn the upsampling kernel",
)
parser.add_argument(
    "--latent_factor",
    type=int,
    default=1,
    help="Full resolution -> 1, other W,H/factor",
)
parser.add_argument("--mod_base", type=int, default=7, help="Number of base")

parser.add_argument(
    "--highest_flag", type=int, default=1, help="Full resolution -> 1, other W,H/factor"
)
parser.add_argument("--context_arm", type=int, default=16, help="8,16,24,32")
parser.add_argument("--dim_arm_mod", type=int, default=16, help="arm dimension")
parser.add_argument("--use_candidate", default=True, help="Use candidate")
parser.add_argument("--type", default="kodak", help="Dateset type")
parser.add_argument("--mod_hid_layer", type=int, default=0, help="3x3 mod layer")

parser.add_argument("--sythesis_features", type=int, default=12, help="hidden")
parser.add_argument("--hidden_features", type=int, default=64, help="hidden")
parser.add_argument("--hidden_layer", type=int, default=2, help="layer")
parser.add_argument("--scale", type=int, default=1, help="Predict every scale*1 pixel")
parser.add_argument(
    "--lambda_rate", type=float, default=1e-3, metavar="LR", help="weight"
)
parser.add_argument(
    "--lambda_rate_list",
    type=float,
    nargs="+",
    default=[6.0e-4, 8.0e-4, 1.5e-3, 2.5e-3, 3.5e-3, 5.0e-3, 7.0e-3, 1.5e-2],
    metavar="LR",
    help="list of lambda weights",
)
parser.add_argument(
    "--start_index", type=int, default=0, help="Predict every scale*1 pixel"
)

parser.add_argument("--mask_type", type=str, default="full")

parser.add_argument("--wsmse_tag", type=int, default=0)

parser.add_argument("--swhdc_tag", type=int, default=0)
parser.add_argument("--swhdc_dilations", type=int, nargs="+", default=[1, 2, 3, 4])
# ERP-aware padding: circular on horizontal axis (0°/360° wrap), replicate on vertical.
# Affects Upsampling and SynthesisResidualLayer(kernel_size=3) inside full_net.
parser.add_argument(
    "--erp_padding",
    type=int,
    default=0,
    help="1 = circular-H + replicate-V padding (ERP images); 0 = replicate all (default)",
)

parser.add_argument(
    "--workdir",
    type=str,
    default=None,
    help="Directory to save CSV results. If None, CSV is not saved.",
)

parser.add_argument("--train_steps_1", type=int, default=100000)
parser.add_argument("--train_steps_2", type=int, default=10000)


args = parser.parse_args()


def save_metrics_to_csv(workdir: str, row: dict):
    """Append a metrics row to {workdir}/results.csv, creating the file and
    header on the first call."""
    os.makedirs(workdir, exist_ok=True)
    csv_path = os.path.join(workdir, "results.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if args.type == "kodak":
    traing_list = range(0, 24)
elif args.type == "clic":
    traing_list = range(0, 41)
elif args.type == "other":
    traing_list = range(0, 30)
    # traing_list = range(0, 1)


all_psnr_list_of_lists = []
all_rate_list_of_lists = []
all_rate_num_list_of_lists = []
eval_all_psnr_list_of_lists = []
eval_all_y_rate_list_of_lists = []
eval_all_mlp_rate_list_of_lists = []
eval_all_y_rate_num_list_of_lists = []
eval_all_mlp_rate_num_list_of_lists = []
eval_all_border_rate_list_of_lists = []
eval_all_border_rate_num_list_of_lists = []
eval_all_total_rate_list_of_lists = []
eval_all_total_rate_num_list_of_lists = []
eval_all_arm_rate_list_of_lists = []
eval_all_arm_rate_num_list_of_lists = []
eval_all_conv_rate_list_of_lists = []
eval_all_conv_rate_num_list_of_lists = []

for num, lambda_rate in enumerate(args.lambda_rate_list):
    seed_everything(1)
    all_psnr = []
    all_rate = []
    all_rate_num = []
    eval_all_psnr = []
    eval_all_y_rate = []
    eval_all_y_rate_num = []
    eval_all_mlp_rate = []
    eval_all_mlp_rate_num = []
    eval_all_rate_arm = []
    eval_all_rate_arm_num = []
    eval_all_rate_conv = []
    eval_all_rate_conv_num = []
    eval_all_border_rate = []
    eval_all_border_rate_num = []
    eval_all_total_rate = []
    eval_all_total_rate_num = []

    args.lambda_rate = lambda_rate

    for it in traing_list:
        idx_str = f"{it + 1:02d}"

        if args.type == "kodak":
            val_folder = f"./dataset/kodak_data_set/kodim{idx_str}"
            lossy_path = f"./dataset/kodak_data_set/kodak_lossy_mask/kodim{idx_str}.png"
            lossyless_path = f"./dataset/kodak_data_set/kodak_mask/kodim{idx_str}.png"
        elif args.type == "clic":
            val_folder = f"./dataset/clic_data_set/clic{idx_str}"
            lossy_path = f"./dataset/clic_data_set/clic_lossy_mask/clic{idx_str}.png"
            lossyless_path = f"./dataset/clic_data_set/clic_mask/clic{idx_str}.png"

        elif args.type == "other":
            val_folder = f"./dataset/other_data_set/othim{idx_str}"
            lossy_path = f"./dataset/other_data_set/other_lossy_mask/othim{idx_str}.png"
            lossyless_path = f"./dataset/other_data_set/other_mask/othim{idx_str}.png"

        args.lambda_rate = lambda_rate
        transform_val = transforms.Compose([transforms.ToTensor()])
        val_dataset = datasets.ImageFolder(val_folder, transform_val)
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        img_in, _ = next(iter(dataloader))
        args.patch_h = img_in.shape[2]
        args.patch_w = img_in.shape[3]
        ifmask = False
        width, height, target_mask_latet = get_mask_h_w(lossy_path)
        target_mask_tensor, target_mask = mm(
            lossy_path, img_in.shape[2], img_in.shape[3]
        )

        if ifmask:
            args.all_pix_num = target_mask_tensor.sum()
        else:
            args.all_pix_num = args.patch_h * args.patch_w
        args.all_pix_num = args.patch_h * args.patch_w
        args.eval_pix_num = args.patch_h * args.patch_w
        print(args)
        folder_path = (
            "./saved/modbase_"
            + str(args.mod_base)
            + "/context_"
            + str(args.context_arm)
            + "_arm_mod_"
            + str(args.dim_arm_mod)
        )
        make_path(folder_path)
        folder_path_ = folder_path + "/sparity_" + str(args.sparsity)
        make_path(folder_path_)
        saved_path = (
            folder_path_
            + "/inr_mod_"
            + str(args.dim_arm_mod)
            + "KODAK_ROI_Global_new_struture_7_"
            + str(args.sythesis_features)
            + "_3_33_orderconcat_operator_"
            + str(args.mod_hid_layer)
            + "_pw_"
            + str(args.lambda_rate)
            + "_img_object"
            + str(it)
            + ".pth"
        )
        total_steps = args.train_steps_1
        total_steps_2 = args.train_steps_2
        steps_til_summary = 1000
        print("top %:", args.sparsity)
        target_mask_flat = target_mask.flatten()
        if args.use_candidate:
            mask_model = train_with_candidates(
                args, target_mask, target_mask_flat, dataloader
            )
        else:
            mask_model = Masked_INR(
                args,
                target_mask,
                sparsity=args.sparsity,
                in_features=2,
                out_features=3 * args.scale * args.scale,
                hidden_features=args.hidden_features,
                hidden_layers=args.hidden_layer,
            )

        print(mask_model)
        mask_model.cuda()
        print("train the", it, "-th image")

        out_psnr, out_rate, rate_num = train(
            target_mask_flat,
            mask_model,
            dataloader,
            total_steps,
            total_steps_2,
            steps_til_summary,
            it,
            saved_path,
        )

        all_psnr.append(out_psnr)
        all_rate.append(out_rate)
        all_rate_num.append(rate_num)
        print(
            "Trained the image with PSNR:",
            out_psnr,
            " latent bits",
            out_rate,
            "latent bits number",
            rate_num,
        )
        print(all_psnr)
        print(all_rate)
        print(all_rate_num)

        checkpoints = torch.load(saved_path)
        mask_model.load_state_dict(checkpoints["model_state_dict"])

        binary_mask = None
        print("load the model:", saved_path, "for the ", it, "-th image")
        mask_model.cuda()
        mask_model.eval()
        (
            eval_out_psnr,
            eval_y_rate,
            eval_y_rate_num,
            eval_network_rate,
            eval_network_rate_num,
            eval_network_rate_arm,
            eval_network_rate_arm_num,
            eval_network_rate_conv,
            eval_network_rate_conv_num,
        ) = eval_model(target_mask_flat, args, mask_model, binary_mask, dataloader, it)

        # --- Compute WS-SSIM ---
        # model_output from eval is discarded; re-run a single forward pass
        # with the loaded (eval-mode) model to get output in (B, H*W, C) format,
        # which is exactly what compute_ws_ssim expects.
        img_in_ssim, _ = next(iter(dataloader))
        H_ssim, W_ssim = img_in_ssim.shape[2], img_in_ssim.shape[3]
        coords_ssim = get_mgrid(W_ssim // args.scale, H_ssim // args.scale, 2).cuda()
        with torch.no_grad():
            model_output_ssim, _, _ = mask_model(coords_ssim)  # (1, H*W, 3)
        pixels_ssim = (
            img_in_ssim.permute(0, 2, 3, 1).reshape(1, H_ssim * W_ssim, 3).cuda()
        )
        eval_wsssim = compute_ws_ssim(
            model_output_ssim, pixels_ssim, H_ssim, W_ssim
        ).item()
        print(f"eval_wsssim: {eval_wsssim:.6f}")
        # -----------------------

        # --- Save decoded image ---
        decoded_img = (
            model_output_ssim.squeeze(0)  # (1, H*W, 3)  # (H*W, 3)
            .reshape(H_ssim, W_ssim, 3)  # (H, W, 3)
            .clamp(0, 1)
            .detach()
            .cpu()
            .numpy()
        )
        decoded_img_uint8 = (decoded_img * 255).astype(np.uint8)

        # cv2 expects BGR
        decoded_img_bgr = cv2.cvtColor(decoded_img_uint8, cv2.COLOR_RGB2BGR)

        if args.workdir is not None:
            decoded_dir = os.path.join(args.workdir, "decoded_images")
            os.makedirs(decoded_dir, exist_ok=True)
            image_name = f"othim{idx_str}"
            decoded_path = os.path.join(
                decoded_dir,
                f"{image_name}_mask{args.mask_type}_wsmse{args.wsmse_tag}_swhdc{args.swhdc_tag}_erp{args.erp_padding}_lambda{lambda_rate}.png",
            )
            cv2.imwrite(decoded_path, decoded_img_bgr)
            print(f"Decoded image saved to {decoded_path}")

        eval_all_psnr.append(eval_out_psnr)
        eval_all_y_rate.append(eval_y_rate)
        eval_all_y_rate_num.append(eval_y_rate_num)
        eval_all_mlp_rate.append(eval_network_rate)
        eval_all_mlp_rate_num.append(eval_network_rate_num)
        eval_all_rate_arm.append(eval_network_rate_arm)
        eval_all_rate_arm_num.append(eval_network_rate_arm_num)
        eval_all_rate_conv.append(eval_network_rate_conv)
        eval_all_rate_conv_num.append(eval_network_rate_conv_num)
        # eval_border_rate_num = get_border_bits(lossyless_path, it)
        eval_border_rate_num = 0
        eval_border_rate = eval_border_rate_num / args.eval_pix_num
        eval_all_border_rate.append(eval_border_rate)
        eval_all_border_rate_num.append(eval_border_rate_num)

        eval_all_rate_y_mlp_latent = [
            y + mlp + border
            for y, mlp, border in zip(
                eval_all_y_rate, eval_all_mlp_rate, eval_all_border_rate
            )
        ]
        eval_all_rate_y_mlp_latent_num = [
            y + mlp + border
            for y, mlp, border in zip(
                eval_all_y_rate_num, eval_all_mlp_rate_num, eval_all_border_rate_num
            )
        ]
        eval_all_total_rate.append(eval_all_rate_y_mlp_latent)
        eval_all_total_rate_num.append(eval_all_rate_y_mlp_latent_num)
        print(
            "Evaluate the image: PSNR:",
            eval_out_psnr,
            "All bits:",
            eval_all_rate_y_mlp_latent[-1],
            " latent bits:",
            eval_y_rate,
            " network bits:",
            eval_network_rate,
        )
        print(
            "Evaluate arm network bits:",
            eval_network_rate_arm,
            "Evaluate synthesis network bits:",
            eval_network_rate_conv,
        )
        print(
            "Image All bits num:",
            eval_all_rate_y_mlp_latent_num[-1],
            "latent bits num:",
            eval_y_rate_num,
            "mlp bits num:",
            eval_network_rate_num,
            "border bits num:",
        )
        print(
            "arm bits num:",
            eval_network_rate_arm_num,
            "synthesis bits num:",
            eval_network_rate_conv_num,
        )
        print(eval_all_psnr)
        print(eval_all_rate_y_mlp_latent)
        print(eval_all_rate_y_mlp_latent_num)

        # ---- CSV logging ----
        if args.workdir is not None:
            if args.type == "kodak":
                image_name = f"kodim{idx_str}"
            elif args.type == "clic":
                image_name = f"clic{idx_str}"
            elif args.type == "other":
                image_name = f"othim{idx_str}"
            else:
                image_name = f"img{idx_str}"

            metrics_row = {
                "image_name": image_name,
                "mask_type": args.mask_type,
                "wsmse_tag": args.wsmse_tag,
                "swhdc_tag": args.swhdc_tag,
                "erp_padding": args.erp_padding,
                "lambda_rate": lambda_rate,
                # Training metrics
                "train_psnr": out_psnr,
                "train_rate_bpp": out_rate,
                "train_rate_bits": rate_num,
                # Evaluation metrics
                "eval_psnr": eval_out_psnr,
                "eval_wsssim": eval_wsssim,
                "eval_y_rate_bpp": eval_y_rate,
                "eval_y_rate_bits": eval_y_rate_num,
                "eval_mlp_rate_bpp": eval_network_rate,
                "eval_mlp_rate_bits": eval_network_rate_num,
                "eval_arm_rate_bpp": eval_network_rate_arm,
                "eval_arm_rate_bits": eval_network_rate_arm_num,
                "eval_conv_rate_bpp": eval_network_rate_conv,
                "eval_conv_rate_bits": eval_network_rate_conv_num,
                "eval_border_rate_bpp": eval_border_rate,
                "eval_border_rate_bits": eval_border_rate_num,
                "eval_total_rate_bpp": eval_all_rate_y_mlp_latent[-1],
                "eval_total_rate_bits": eval_all_rate_y_mlp_latent_num[-1],
                "total_steps_1": args.train_steps_1,
                "total_steps_2": args.train_steps_2,
            }
            save_metrics_to_csv(args.workdir, metrics_row)
            print(f"Metrics saved to {os.path.join(args.workdir, 'results.csv')}")
        print(
            "Current eval Ave PSNR:",
            np.mean(eval_all_psnr),
            "Ave Bits",
            np.mean(eval_all_rate_y_mlp_latent),
        )
    all_psnr_list_of_lists.append(all_psnr)
    all_rate_list_of_lists.append(all_rate)
    all_rate_num_list_of_lists.append(all_rate_num)
    eval_all_psnr_list_of_lists.append(eval_all_psnr)
    eval_all_y_rate_list_of_lists.append(eval_all_y_rate)
    eval_all_y_rate_num_list_of_lists.append(eval_all_y_rate_num)
    eval_all_mlp_rate_list_of_lists.append(eval_all_mlp_rate)
    eval_all_mlp_rate_num_list_of_lists.append(eval_all_mlp_rate_num)
    eval_all_border_rate_list_of_lists.append(eval_all_border_rate)
    eval_all_border_rate_num_list_of_lists.append(eval_all_border_rate_num)
    eval_all_total_rate_list_of_lists.append(eval_all_total_rate)
    eval_all_total_rate_num_list_of_lists.append(eval_all_total_rate_num)
    eval_all_arm_rate_list_of_lists.append(eval_all_rate_arm)
    eval_all_arm_rate_num_list_of_lists.append(eval_all_rate_arm_num)
    eval_all_conv_rate_list_of_lists.append(eval_all_rate_conv)
    eval_all_conv_rate_num_list_of_lists.append(eval_all_rate_conv_num)

    print(".......Copmlete all dataset training......")
    print(
        "Ave Training PSNR:", np.mean(all_psnr), "Ave Training Bits", np.mean(all_rate)
    )

    print("Training PSNR:", all_psnr)
    print("Training rate:", all_rate)

    print(
        "Evalutation: Ave Eval PSNR:",
        np.mean(eval_all_psnr),
        "Ave Eval all bits:",
        np.mean(eval_all_rate_y_mlp_latent),
        "Ave Eval Bits",
        np.mean(eval_all_y_rate),
        "Ave Eval Network",
        np.mean(eval_all_mlp_rate),
    )
    print("Eval All PSNR:", eval_all_psnr)
    print("Eval All rate", eval_all_rate_y_mlp_latent)
    print("Eval Latent rate", eval_all_y_rate)
    print("Eval MLP rate", eval_all_mlp_rate)

print("======== ALL Results ========")
for i, lambda_rate in enumerate(args.lambda_rate_list):
    print(f"Lambda = {lambda_rate}:")
    print("  all_psnr:", all_psnr_list_of_lists[i])
    print("  all_rate:", all_rate_list_of_lists[i])
    print("  all_rate_num:", all_rate_num_list_of_lists[i])
    print("  eval_all_psnr:", eval_all_psnr_list_of_lists[i])
    print("  eval_all_y_rate:", eval_all_y_rate_list_of_lists[i])
    print("  eval_all_y_rate_num:", eval_all_y_rate_num_list_of_lists[i])
    print("  eval_all_mlp_rate:", eval_all_mlp_rate_list_of_lists[i])
    print("  eval_all_mlp_rate_num:", eval_all_mlp_rate_num_list_of_lists[i])
    print("  eval_all_arm_rate:", eval_all_arm_rate_list_of_lists[i])
    print("  eval_all_arm_rate_num:", eval_all_arm_rate_num_list_of_lists[i])
    print("  eval_all_conv_rate:", eval_all_conv_rate_list_of_lists[i])
    print("  eval_all_conv_rate_num:", eval_all_conv_rate_num_list_of_lists[i])
    print("  eval_all_border_rate:", eval_all_border_rate_list_of_lists[i])
    print("  eval_all_border_rate_num:", eval_all_border_rate_num_list_of_lists[i])
    print("  eval_all_total_rate", eval_all_total_rate_list_of_lists[i])
    print("  eval_all_total_rate_num", eval_all_total_rate_num_list_of_lists[i])
    print("---------------------------------")
