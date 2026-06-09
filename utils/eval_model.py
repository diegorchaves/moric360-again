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


def compute_ws_mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    height, width, _ = img1.shape

    # Utiliza o mesmo dtype e device das imagens de entrada para evitar conflitos (ex: GPU)
    dtype = img1.dtype
    device = img1.device

    phis = torch.arange(height + 1, dtype=dtype, device=device) * torch.pi / height
    deltaTheta = 2 * torch.pi / width

    # Vetorização: substitui o loop for pela diferença direta de slices do tensor
    column = deltaTheta * (-torch.cos(phis[1:]) + torch.cos(phis[:-1]))

    # Adiciona dimensões para (H, 1, 1). O broadcasting do PyTorch aplicará
    # esses pesos automaticamente para as dimensões de Largura (W) e Canais (C).
    w = column.view(height, 1, 1)

    # Cálculo vetorizado e com broadcasting
    mse = ((img1 - img2) ** 2 * w).mean(dim=2)
    wmse = mse.sum() / (4 * torch.pi)

    return wmse


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

    if args.wsmse_tag == 1:
        criterion = compute_ws_mse
    elif args.wsmse_tag == 0:
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
        eval_o = loss_to_psnr(loss_mse_o.item())
        print("eval_object_psnr:", eval_o)

        # 3. AVALIAÇÃO DO FUNDO (BACKGROUND)
        # Copiamos as imagens e forçamos o objeto a ser idêntico (erro = 0 no objeto)
        out_bg = out_full.clone()
        target_bg = target_full.clone()
        out_bg[mask_2d] = target_bg[mask_2d]

        loss_mse_b = criterion(out_bg, target_bg)

        out_bg = out_full.clone()
        target_bg = target_full.clone()
        out_bg[mask_2d] = target_bg[mask_2d]

        loss_mse_b = criterion(out_bg, target_bg)

        psnr_eval = loss_to_psnr(loss_mse.item())
        psnr_eval_o = loss_to_psnr(loss_mse_o.item())
        psnr_eval_b = loss_to_psnr(loss_mse_b.item())
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
