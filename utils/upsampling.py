import math
from typing import List, OrderedDict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class UpsamplingConvTranspose2d(nn.Module):
   

    kernel_bilinear = torch.tensor(
        [
            [0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625],
        ]
    )

    kernel_bicubic = torch.tensor(
        [
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
        ]
    )


    def __init__(
        self,
        upsampling_kernel_size: int,
        static_upsampling_kernel: bool
    ):
       
        super().__init__()

        assert upsampling_kernel_size >= 4, (
            f"Upsampling kernel size should be >= 4." f"Found {upsampling_kernel_size}"
        )

        assert upsampling_kernel_size % 2 == 0, (
            f"Upsampling kernel size should be even." f"Found {upsampling_kernel_size}"
        )

        self.upsampling_kernel_size = upsampling_kernel_size
        self.static_upsampling_kernel = static_upsampling_kernel

       
        self.weight = nn.Parameter(
            torch.empty(1, 1, upsampling_kernel_size, upsampling_kernel_size),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.empty((1)), requires_grad=True)
        self.initialize_parameters()
        
        if self.static_upsampling_kernel:
            
            self.register_buffer("static_kernel", self.weight.data.clone(), persistent=False)
        else:
            self.static_kernel = None

    def initialize_parameters(self) -> None:
       
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

       
        K = self.upsampling_kernel_size
        self.upsampling_padding = (K // 2, K // 2, K // 2, K // 2)
        self.upsampling_crop = (3 * K - 2) // 2

        if K < 8:
            kernel_init = UpsamplingConvTranspose2d.kernel_bilinear
        else:
            kernel_init = UpsamplingConvTranspose2d.kernel_bicubic

       
        tmpad = (K - kernel_init.size()[0]) // 2
        upsampling_kernel = F.pad(
            kernel_init.clone().detach(),
            (tmpad, tmpad, tmpad, tmpad),
            mode="constant",
            value=0.0,
        )

       
        upsampling_kernel = rearrange(upsampling_kernel, "k_h k_w -> 1 1 k_h k_w")
        self.weight = nn.Parameter(upsampling_kernel, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
       
        upsampling_weight = (
            self.static_kernel if self.static_upsampling_kernel else self.weight
        )

        x_pad = F.pad(x, self.upsampling_padding, mode="replicate")
        y_conv = F.conv_transpose2d(x_pad, upsampling_weight, stride=2, output_padding=1)

       
        H, W = y_conv.size()[-2:]
        results = y_conv[
            :,
            :,
            self.upsampling_crop : H - self.upsampling_crop,
            self.upsampling_crop : W - self.upsampling_crop,
        ]

        return results


class SphericalUpsamplingConvTranspose2d(nn.Module):
    """
    2× upsampling ciente da geometria ERP (Equirectangular Projection).

    Diferenças em relação a UpsamplingConvTranspose2d:

    1. **Padding horizontal circular** — a borda esquerda e direita do ERP
       correspondem à mesma longitude (0 = 2π), portanto o padding usa
       ``mode="circular"`` nessa direção. Verticalmente usa ``mode="replicate"``
       (valores dos polos não se enrolam).

    2. **Kernel inicializado com distâncias geodésicas** (Catmull-Rom esférico
       para K ≥ 8, bilinear esférico para K < 8). Para cada posição (i, j) do
       kernel K×K a distância esférica normalizada pelo espaçamento vertical é::

           d(tv, th) = sqrt(tv² + cos²(φ_ref) · th²)

       onde tv = (i − K/2 + 0.5) / 2 e th = (j − K/2 + 0.5) / 2 são os
       offsets fraccionais no espaço de entrada e φ_ref é uma latitude de
       referência (default: 30° = π/6).

       - φ_ref = 0  (equador): cos = 1  → isotrópico → equivalente ao plano.
       - φ_ref > 0           : cos < 1  → pixels horizontais geodesicamente
                               mais próximos → kernel não-separável.

    3. O kernel permanece **learnable** (``nn.Parameter``) — a rede afina os
       coeficientes durante o overfitting sobre a imagem 360°.

    Parâmetros
    ----------
    upsampling_kernel_size : int
        Tamanho K do kernel (par e ≥ 4). Default do pipeline: 8.
    static_upsampling_kernel : bool
        Se True, kernel fixo (congelado); se False (default), aprendível.
    phi_ref : float
        Latitude de referência em radianos para inicialização do kernel esférico.
        Default: math.pi / 6 (30°). Usar 0 recupera o comportamento isotrópico.
    """

    @staticmethod
    def _bilinear_1d(t: float) -> float:
        return max(0.0, 1.0 - abs(t))

    @staticmethod
    def _catmull_rom(t: float) -> float:
        t = abs(t)
        if t <= 1.0:
            return 1.25 * t ** 3 - 2.25 * t ** 2 + 1.0
        elif t <= 2.0:
            return -0.75 * t ** 3 + 3.75 * t ** 2 - 6.0 * t + 3.0
        return 0.0

    @classmethod
    def _compute_spherical_kernel(
        cls,
        K: int,
        phi_ref: float,
    ) -> torch.Tensor:
        cos_phi = math.cos(phi_ref)
        interp_fn = cls._catmull_rom if K >= 8 else cls._bilinear_1d

        kernel = torch.zeros(K, K, dtype=torch.float32)
        for i in range(K):
            tv = (i - K / 2 + 0.5) / 2.0
            for j in range(K):
                th = (j - K / 2 + 0.5) / 2.0
                d = math.sqrt(tv ** 2 + (cos_phi * th) ** 2)
                kernel[i, j] = interp_fn(d)
        return kernel

    def __init__(
        self,
        upsampling_kernel_size: int,
        static_upsampling_kernel: bool,
        phi_ref: float = math.pi / 6,
    ):
        super().__init__()

        assert upsampling_kernel_size >= 4, (
            f"Upsampling kernel size should be >= 4. Found {upsampling_kernel_size}"
        )
        assert upsampling_kernel_size % 2 == 0, (
            f"Upsampling kernel size should be even. Found {upsampling_kernel_size}"
        )

        self.upsampling_kernel_size = upsampling_kernel_size
        self.static_upsampling_kernel = static_upsampling_kernel
        self.phi_ref = phi_ref

        self.weight = nn.Parameter(
            torch.empty(1, 1, upsampling_kernel_size, upsampling_kernel_size),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.empty(1), requires_grad=True)
        self.initialize_parameters()

        if self.static_upsampling_kernel:
            self.register_buffer(
                "static_kernel", self.weight.data.clone(), persistent=False
            )
        else:
            self.static_kernel = None

    def initialize_parameters(self) -> None:
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

        K = self.upsampling_kernel_size
        self.upsampling_crop = (3 * K - 2) // 2

        spherical_kernel = self._compute_spherical_kernel(K, self.phi_ref)
        upsampling_kernel = rearrange(spherical_kernel, "k_h k_w -> 1 1 k_h k_w")
        self.weight = nn.Parameter(upsampling_kernel, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        upsampling_weight = (
            self.static_kernel if self.static_upsampling_kernel else self.weight
        )

        K = self.upsampling_kernel_size
        pad = K // 2

        x_pad = F.pad(x, (pad, pad, 0, 0), mode="circular")
        x_pad = F.pad(x_pad, (0, 0, pad, pad), mode="replicate")

        y_conv = F.conv_transpose2d(x_pad, upsampling_weight, stride=2, output_padding=1)

        H, W = y_conv.size()[-2:]
        crop = self.upsampling_crop
        results = y_conv[:, :, crop : H - crop, crop : W - crop]
        return results

    def reinitialize_parameters(self) -> None:
        self.initialize_parameters()


class Upsampling(nn.Module):
   

    def __init__(
        self,
        upsampling_kernel_size: int,
        static_upsampling_kernel: bool,
        highest_flag: int = 1,
        erp_mode: bool = False,
        phi_ref: float = math.pi / 6,
    ):
        super().__init__()

        self.highest_flag = highest_flag

        if erp_mode:
            self.conv_transpose2d = SphericalUpsamplingConvTranspose2d(
                upsampling_kernel_size, static_upsampling_kernel, phi_ref
            )
        else:
            self.conv_transpose2d = UpsamplingConvTranspose2d(
                upsampling_kernel_size, static_upsampling_kernel
            )

    def forward(self, decoder_side_latent: List[Tensor], masks: List[Tensor]) -> Tensor:
      
        latent_reversed = (decoder_side_latent)

        upsampled_latent = latent_reversed[0] 
        masks = None
        if masks == None:
           
            for target_tensor in latent_reversed[1:]:
                
                x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
               

                x = self.conv_transpose2d(x)
                x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])

               
                x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]
                
                upsampled_latent = torch.cat((target_tensor, x), dim=1)
        else:
            upsampled_mask = masks[0]
            for target_tensor,target_mask in zip(latent_reversed[1:], masks[1:]):
                
                x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
               

                x = self.conv_transpose2d(x)
                x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])

              
                x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]
               
                upsampled_latent = torch.cat((target_tensor, x), dim=1)
        if self.highest_flag==0:
            x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
            x = self.conv_transpose2d(x)
            upsampled_latent = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])
           
        return upsampled_latent

    def get_param(self) -> OrderedDict[str, Tensor]:
        
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]):
        
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
       
        self.conv_transpose2d.initialize_parameters()
