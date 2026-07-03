"""
profile_model.py
----------------
Mede métricas de custo computacional dos modelos:
  - GFLOPs / GMACs  (via torch.profiler — funciona com dynamic masking)
  - Número de parâmetros e tamanho em MB  (via fvcore parameter_count_table)
  - Tempo de inferência médio (com warmup, CUDA events)

Obs: fvcore.FlopCountAnalysis usa tracing ONNX estático e NÃO funciona com
modelos que fazem indexação booleana dinâmica (como este Masked_INR).
Por isso usamos torch.profiler(with_flops=True) para contar FLOPs.

Uso:
    python profile_model.py --hidden_features 64 --hidden_layer 2 --scale 1
    python profile_model.py --hidden_features 64 --hidden_layer 2 --scale 1 --verbose_flops
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import parameter_count_table


def get_mgrid(w_sidelen, h_sidelen, dim=2):
    """Grade de coordenadas normalizadas [-1, 1] — identica a train.py."""
    x = torch.linspace(-1, 1, steps=w_sidelen)
    y = torch.linspace(-1, 1, steps=h_sidelen)
    mgrid = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1)
    mgrid = mgrid.unsqueeze(0).permute(0, 3, 2, 1)
    return mgrid



# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def model_size_mb(model: torch.nn.Module) -> float:
    """Retorna o tamanho total do modelo (parâmetros + buffers) em MB."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 ** 2)


def count_parameters(model: torch.nn.Module) -> int:
    """Conta apenas parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(
    model: torch.nn.Module,
    dummy_input,
    n_warmup: int = 50,
    n_runs: int = 200,
    device: str = "cuda",
) -> dict:
    """
    Mede o tempo de inferência com warmup adequado.
    Usa CUDA events para medição precisa em GPU, ou time.perf_counter em CPU.
    """
    model.eval()
    times = []

    with torch.no_grad():
        # ── Warmup ──────────────────────────────────────────────────────────
        for _ in range(n_warmup):
            if isinstance(dummy_input, (tuple, list)):
                model(*dummy_input)
            else:
                model(dummy_input)

        # ── Medição ─────────────────────────────────────────────────────────
        if device == "cuda" and torch.cuda.is_available():
            for _ in range(n_runs):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                if isinstance(dummy_input, (tuple, list)):
                    model(*dummy_input)
                else:
                    model(dummy_input)
                end_ev.record()
                torch.cuda.synchronize()
                times.append(start_ev.elapsed_time(end_ev))  # ms
        else:
            for _ in range(n_runs):
                t0 = time.perf_counter()
                if isinstance(dummy_input, (tuple, list)):
                    model(*dummy_input)
                else:
                    model(dummy_input)
                times.append((time.perf_counter() - t0) * 1000)  # ms

    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms":  float(arr.std()),
        "min_ms":  float(arr.min()),
        "max_ms":  float(arr.max()),
        "fps":     1000.0 / float(arr.mean()),
    }


def measure_flops(model: torch.nn.Module, dummy_input, device: str = "cuda") -> dict:
    """
    Conta FLOPs usando hooks em nn.Linear e nn.Conv2d.
    Funciona com qualquer modelo PyTorch independente de dynamic masking.
    GMACs = GFLOPs / 2  (cada MAC = multiplicacao + adicao = 2 FLOPs)
    """
    total_flops = [0]

    def linear_hook(module, inp, out):
        # FLOPs = 2 * in_features * out_features * batch_elements
        batch = inp[0].numel() // module.in_features
        total_flops[0] += 2 * module.in_features * module.out_features * batch

    def conv2d_hook(module, inp, out):
        # FLOPs = 2 * Cin * Kh * Kw * Hout * Wout * (Cout / groups) * batch
        batch, _, h_out, w_out = out.shape
        k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        flops = (2 * module.in_channels * k_h * k_w * h_out * w_out
                 * (module.out_channels // module.groups) * batch)
        total_flops[0] += flops

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))

    model.eval()
    with torch.no_grad():
        if isinstance(dummy_input, (tuple, list)):
            model(*dummy_input)
        else:
            model(dummy_input)

    for h in hooks:
        h.remove()

    tf = total_flops[0]
    return {
        "GFLOPs": tf / 1e9,
        "GMACs":  tf / 2e9,
    }



def print_summary(
    model_name: str,
    flop_info: dict,
    timing_info: dict,
    size_mb: float,
    n_params: int,
):
    sep = "─" * 60
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    print(f"\nParametros & Tamanho")
    print(sep)
    print(f"  Parametros treinaveis : {n_params:,}")
    print(f"  Tamanho do modelo     : {size_mb:.4f} MB")

    print(f"\nCusto Computacional (torch.profiler)")
    print(sep)
    print(f"  GFLOPs  : {flop_info['GFLOPs']:.4f}")
    print(f"  GMACs   : {flop_info['GMACs']:.4f}")

    print(f"\nTempo de Inferencia")
    print(sep)
    print(f"  Media   : {timing_info['mean_ms']:.3f} ms")
    print(f"  Desvio  : {timing_info['std_ms']:.3f} ms")
    print(f"  Minimo  : {timing_info['min_ms']:.3f} ms")
    print(f"  Maximo  : {timing_info['max_ms']:.3f} ms")
    print(f"  FPS     : {timing_info['fps']:.1f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profiling de modelos com fvcore")

    # Hiperparametros do Masked_INR
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--hidden_layer",    type=int, default=2)
    parser.add_argument("--scale",           type=int, default=1)
    parser.add_argument("--sparsity",        type=float, default=0.0)

    # Resolucao da imagem de entrada (H x W)
    parser.add_argument("--height", type=int, default=512, help="Altura da imagem")
    parser.add_argument("--width",  type=int, default=1024,  help="Largura da imagem")

    # Controle
    parser.add_argument("--device",   type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_runs",   type=int, default=200)
    parser.add_argument(
        "--verbose_flops", action="store_true",
        help="Imprime tabela detalhada de FLOPs por camada"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\nDevice: {device}")
    print(f"Resolucao: {args.height}x{args.width}  |  "
          f"hidden={args.hidden_features}  layers={args.hidden_layer}  scale={args.scale}")

    # ── Completar args com todos os defaults que Masked_INR espera ──────────
    # Espelha os defaults de train.py para nao dar AttributeError.
    # Estes sao os valores usados na configuracao dos experimentos ativos:
    #   mask=erp, loss=combined, swhdc_tag=1, erp_padding=1
    _model_defaults = {
        # Upsampling
        "local_upsampling_kernel_size": 8,
        "upsampling_kernel_size":       8,
        "static_upsampling_kernel":     False,
        "highest_flag":                 1,
        "latent_factor":                1,
        # ARM
        "context_arm": 16,
        "dim_arm_mod": 16,
        # Modulation / synthesis
        "mod_base":          7,
        "mod_hid_layer":     0,
        "sythesis_features": 12,
        # SWHDC / ERP  ← configuracao dos experimentos ativos
        "swhdc_tag":      1,
        "swhdc_dilations": [1, 2, 3, 4],
        "erp_padding":    1,
        # batch_size e usado no __init__ para alocar os latentes
        "batch_size": 1,
        # Loss (nao afeta arquitetura)
        "loss_type":        "combined",
        "lambda_rate":      1e-3,
        "lambda_rate_list": [6e-4, 8e-4],
    }
    for k, v in _model_defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # ── Importar o modelo ─────────────────────────────────────────────────────
    from models.model import Masked_INR

    H, W = args.height, args.width

    # target_mask: shape (1, 1, H, W) — igual ao target_mask_4d do train.py.
    # Com mask_type=erp: polares (top 25% e bottom 25%) = True, equatorial = False.
    target_mask = torch.ones(1, 1, H, W, dtype=torch.bool)
    top    = int(H * 0.25)
    bottom = int(H * 0.75)
    target_mask[:, :, top:bottom, :] = False
    target_mask = target_mask.to(device)

    model = Masked_INR(
        args,
        target_mask,
        sparsity=args.sparsity,
        in_features=2,
        out_features=3 * args.scale * args.scale,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layer,
    ).to(device)
    model.eval()

    # ── Coordenadas: exatamente como get_mgrid(W, H, 2) do train.py ──────────
    # get_mgrid retorna (1, 2, H, W) — e train.py passa esse tensor direto para model().
    dummy_coords = get_mgrid(W // args.scale, H // args.scale, 2).to(device)


    # ── Calcular metricas ─────────────────────────────────────────────────────
    size_mb  = model_size_mb(model)
    n_params = count_parameters(model)

    print("\nCalculando FLOPs...")
    flop_info = measure_flops(model, dummy_coords, device=args.device)

    print("Medindo tempo de inferencia...")
    timing_info = measure_inference_time(
        model, dummy_coords,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        device=args.device,
    )

    # ── Exibir resumo ─────────────────────────────────────────────────────────
    print_summary("Masked_INR", flop_info, timing_info, size_mb, n_params)

    # ── Tabela detalhada por camada (opcional) ────────────────────────────────
    if args.verbose_flops:
        print("\nTabela de parametros (fvcore):")
        print(parameter_count_table(model, max_depth=4))


if __name__ == "__main__":
    main()
