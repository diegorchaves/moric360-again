import torch
import torchvision.transforms as transforms
from PIL import Image

import sys
import os

# Adiciona o diretório principal do projeto (moric360-again) ao path do Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa a função diretamente do seu arquivo
from utils.eval_model import compute_ws_psnr

def load_and_convert_to_png(img_path):
    """Verifica se a imagem é .bmp, se for, converte para .png e salva no disco."""
    if img_path.lower().endswith('.bmp'):
        new_path = img_path.rsplit('.', 1)[0] + '.png'
        print(f"Convertendo '{img_path}' para '{new_path}'...")
        img = Image.open(img_path).convert("RGB")
        img.save(new_path, format="PNG")
        return new_path
    return img_path

def calcular_ws_psnr_standalone(img_path1, img_path2):
    # Converte para .png no disco se for .bmp
    img_path1 = load_and_convert_to_png(img_path1)
    img_path2 = load_and_convert_to_png(img_path2)

    # 1. Carrega as duas imagens usando Pillow
    img1_pil = Image.open(img_path1).convert("RGB")
    img2_pil = Image.open(img_path2).convert("RGB")
    
    # 2. Converte as imagens para tensores PyTorch
    # O transforms.ToTensor() converte a imagem para o shape (Canais, Altura, Largura) 
    # e já normaliza os valores dos pixels para a escala de [0.0, 1.0]
    transform = transforms.ToTensor()
    img1_tensor = transform(img1_pil)
    img2_tensor = transform(img2_pil)
    
    # 3. A função compute_ws_psnr espera o shape no formato (Altura, Largura, Canais)
    # Por isso usamos o permute(1, 2, 0) para alterar as dimensões
    img1_tensor = img1_tensor.permute(1, 2, 0)
    img2_tensor = img2_tensor.permute(1, 2, 0)
    
    # 4. Chama a função 
    # (Como usamos ToTensor que colocou as imagens entre 0 e 1, o max_val=1.0 está correto)
    valor_ws_psnr = compute_ws_psnr(img1_tensor, img2_tensor, max_val=1.0)
    
    print(f"WS-PSNR: {valor_ws_psnr:.6f} dB")

if __name__ == "__main__":
    # Substitua pelos caminhos reais das suas imagens
    imagem_original = "/home/diego/Desktop/moric360-again/imgs_zoom_final/Landing_li2.bmp"
    imagem_reconstruida = "/home/diego/Desktop/moric360-again/imgs_zoom_final/othim24_maskerp_wsmse0_swhdc1_erp1_lambda0.065.png"
    
    calcular_ws_psnr_standalone(imagem_original, imagem_reconstruida)
