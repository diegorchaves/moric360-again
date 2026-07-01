import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.quantizemodel import quantize_model

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def weights(H, W, device):

    phis = torch.arange(H + 1, device=device) * torch.pi / H
    deltaTheta = 2 * torch.pi / W

    column = deltaTheta * (torch.cos(phis[:-1]) - torch.cos(phis[1:]))

    w = column.view(1, H, 1, 1)

    return w


def compute_ws_ssim(img1, img2, H, W, K1=0.01, K2=0.03, L=1.0):
    """
    Calcula o Weighted Spherical SSIM (WS-SSIM) totalmente otimizado em PyTorch.
    Suporta tensores no formato (batch_size, H * W, C) vindos da GPU.
    """
    B, _, C = img1.shape
    device = img1.device

    # Redimensiona de (B, H*W, C) para o formato padrão do torch (B, C, H, W)
    img1_2d = img1.view(B, H, W, C).permute(0, 3, 1, 2)
    img2_2d = img2.view(B, H, W, C).permute(0, 3, 1, 2)

    # Filtro gaussiano
    k = 11
    sigma = 1.5
    pad = k // 2

    # Cria a janela gaussiana na gpu
    coords = torch.arange(k, dtype=torch.float32, device=device) - pad
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    window = torch.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigma**2))
    window = (window / window.sum()).view(1, 1, k, k).expand(C, 1, k, k)

    w_base = weights(H, W, device)  # Shape: (1, H, 1, 1)
    W_2d = w_base.view(H, 1).expand(H, W)
    Wi = W_2d[pad:-pad, pad:-pad]
    weight_sum = Wi.sum()

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = F.conv2d(img1_2d, window, groups=C)
    mu2 = F.conv2d(img2_2d, window, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_2d * img1_2d, window, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2_2d * img2_2d, window, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1_2d * img2_2d, window, groups=C) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    # Aplicação dos pesos esféricos
    Wi_expanded = Wi.view(1, 1, Wi.shape[0], Wi.shape[1])
    weighted_ssim = ssim_map * Wi_expanded

    ssim_per_channel = torch.sum(weighted_ssim, dim=(2, 3)) / weight_sum

    return torch.mean(ssim_per_channel)


def compute_ws_mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:

    height, width, _ = img1.shape

    dtype = img1.dtype
    device = img1.device

    phis = torch.arange(height + 1, dtype=dtype, device=device) * torch.pi / height
    deltaTheta = 2 * torch.pi / width

    # Pesos por linha: shape (H,)
    column = deltaTheta * (-torch.cos(phis[1:]) + torch.cos(phis[:-1]))

    # Expande para (H, 1, 1) — broadcasting cobre W e C automaticamente
    w = column.view(height, 1, 1)

    # Soma sobre H e W, mantém canais (C) separados → shape (C,)
    # Em seguida normaliza por 4π e tira a média dos canais
    wmse_per_channel = ((img1 - img2) ** 2 * w).sum(dim=(0, 1)) / (4 * torch.pi)
    return wmse_per_channel.mean()


def compute_ws_psnr(
    img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0
) -> float:

    # Converte para numpy float64 (igual à referência: float64(img))
    img1 = img1.detach().cpu().numpy().astype(np.float64)
    img2 = img2.detach().cpu().numpy().astype(np.float64)

    height, width = img1.shape[0], img1.shape[1]

    # Matriz de pesos esféricos (H, W)
    phis = np.arange(height + 1) * np.pi / height
    deltaTheta = 2 * np.pi / width
    column = np.asarray(
        [deltaTheta * (-np.cos(phis[j + 1]) + np.cos(phis[j])) for j in range(height)]
    )
    w = np.repeat(column[:, np.newaxis], width, axis=1)  # (H, W)
    w_expanded = w[:, :, np.newaxis]  # (H, W, 1)

    # WS-MSE por canal: soma sobre H e W, normaliza por 4pi -> shape (C,)
    squared_diff = (img1 - img2) ** 2
    weighted_squared_diff = squared_diff * w_expanded
    wmse_three_channel = np.sum(np.sum(weighted_squared_diff, axis=0), axis=0) / (
        4 * np.pi
    )

    # Evita log(0)
    wmse_three_channel = np.where(wmse_three_channel == 0, 1e-10, wmse_three_channel)

    # PSNR por canal, depois média
    wspsnr_three_channel = 10 * np.log10(max_val**2 / wmse_three_channel)
    return float(np.mean(wspsnr_three_channel))


def loss_to_psnr(loss, max=1):
    return 10 * np.log10(max**2 / np.asarray(loss))


def compute_model_rate(model):
    rate_mlp = 0.0
    rate_arm = 0.0
    rate_conv = 0.0
    rate_per_module = model.get_network_rate()
    for model_name, module_rate in rate_per_module.items():
        for _, param_rate in module_rate.items():  # weight, bias
            if model_name == "arm":
                rate_arm += param_rate
            elif model_name == "conv_mod":
                rate_conv += param_rate
            rate_mlp += param_rate
    return rate_mlp, rate_arm, rate_conv


def get_mgrid(w_sidelen, h_sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    x = torch.linspace(-1, 1, steps=w_sidelen)
    y = torch.linspace(-1, 1, steps=h_sidelen)
    tensors = (x, y) if dim == 2 else (x,) * dim

    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)

    mgrid = mgrid.unsqueeze(0).permute(0, 3, 2, 1)

    return mgrid


def eval_model(target_mask, args, model, binary_mask, dataloader, img_index):

    loss_type = getattr(args, "loss_type", None)
    if loss_type == "combined":
        from utils.combined_loss import compute_combined_loss
        _ld = getattr(args, "lambda_dist", 0.75 * 0.1 * (2.0 ** -5))
        _lp = getattr(args, "lambda_percep", 1.0)
        criterion = lambda i1, i2, ld=_ld, lp=_lp: compute_combined_loss(i1, i2, ld, lp)[0]
    elif loss_type == "wsmse" or getattr(args, "wsmse_tag", 0) == 1:
        criterion = compute_ws_mse
    else:
        criterion = nn.MSELoss().cuda()

    for batch_idx, (img_in, _) in enumerate(dataloader, 0):
        batch_size, _, height, width = img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size, -1, 3).cuda()
        pixels1 = pixels[:, target_mask, :]
        pixels2 = pixels[:, ~target_mask, :]

        coords = get_mgrid(width, height, 2).cuda()
        print("********************Evalutation with quantization")
        print("********************Starting quantizing models")
        model_q = quantize_model(model, binary_mask, coords, pixels, args)
        model = model_q

        torch.cuda.empty_cache()

        model.eval()
        model_output, rate, binary_mask = model(coords, binary_mask)

        img_out = model_output.view(batch_size, height, width, 3).permute(0, 3, 1, 2)
        # --- Salvar aqui a img_decoded ---

        bits_rate_eval = rate.sum() / (args.eval_pix_num)
        bits_rate_eval_num = rate.sum()
        # loss_mse = criterion(model_output, pixels)
        #
        out_full = model_output.squeeze(0).view(height, width, 3)
        target_full = pixels.squeeze(0).view(height, width, 3)
        loss_mse = criterion(out_full, target_full)

        # loss_mse_o = criterion(model_output[:, target_mask, :], pixels1)
        # loss_mse_b = criterion(model_output[:, ~target_mask, :], pixels2)
        #
        # 1. Transformar a máscara achatada de volta para 2D (Altura, Largura)
        mask_2d = target_mask.view(height, width)

        # 2. AVALIAÇÃO DO OBJETO
        # Copiamos as imagens e forçamos o fundo a ser idêntico (erro = 0 no fundo)
        out_obj = out_full.clone()
        target_obj = target_full.clone()
        out_obj[~mask_2d] = target_obj[~mask_2d]

        loss_mse_o = criterion(out_obj, target_obj)

        # 3. AVALIAÇÃO DO FUNDO (BACKGROUND)
        # Copiamos as imagens e forçamos o objeto a ser idêntico (erro = 0 no objeto)
        out_bg = out_full.clone()
        target_bg = target_full.clone()
        out_bg[mask_2d] = target_bg[mask_2d]

        loss_mse_b = criterion(out_bg, target_bg)

        # PSNR sempre calculado como WS-PSNR para fins de comparação justa
        psnr_eval = compute_ws_psnr(out_full, target_full)
        psnr_eval_o = compute_ws_psnr(out_obj, target_obj)
        psnr_eval_b = compute_ws_psnr(out_bg, target_bg)
        print("eval_object_psnr:", psnr_eval_o)
        print("full_image_psnr:", psnr_eval)
        print("object_psnr:", psnr_eval_o)
        print("background_psnr:", psnr_eval_b)
        out_network_rate, out_network_rate_arm, out_network_rate_conv = (
            compute_model_rate(model)
        )
        out_network_rate /= args.eval_pix_num
        out_network_rate_arm /= args.eval_pix_num
        out_network_rate_conv /= args.eval_pix_num
        out_network_rate_num, out_network_rate_arm_num, out_network_rate_conv_num = (
            compute_model_rate(model)
        )

        print(
            "********************Evaluation the Image %d-th, BEST PSNR: %0.6f, Print rate %0.6f, Network rate %0.6f. *************************"
            % (img_index, psnr_eval, bits_rate_eval.item(), out_network_rate)
        )
        # img_out=model_output.view(batch_size,height,width,3).permute(0,3,1,2)
        # vutils.save_image(img_out,'./eval_'+str(img_index)+'.png',nrow=1)
        torch.cuda.empty_cache()

    return (
        psnr_eval,
        bits_rate_eval.item(),
        bits_rate_eval_num.item(),
        out_network_rate.item(),
        out_network_rate_num.item(),
        out_network_rate_arm.item(),
        out_network_rate_arm_num.item(),
        out_network_rate_conv.item(),
        out_network_rate_conv_num.item(),
    )


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2.0 * np.pi * x) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return embedding
