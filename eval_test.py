import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from openunmix import model as umx_model
from torch.utils.data import DataLoader
from src.dataset import MUSDBDataset
from src.losses import MultiResolutionSTFTLoss

# ==========================================
# 1. DEFINICIÓN DE MODELOS
# ==========================================
def get_unet_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=2,
        activation='sigmoid'
    )

def get_openunmix_model(n_fft=2048):
    nb_bins = n_fft // 2 + 1
    return umx_model.OpenUnmix(
        nb_bins=nb_bins,
        nb_channels=2,
        hidden_size=512,
        max_bin=int(nb_bins),
    )

# ==========================================
# 2. FUNCIÓN PARA GENERAR GRÁFICO COMPARATIVO
# ==========================================
def save_comparison_chart(stem, unet_loss, umx_loss, out_dir):
    """Genera un único gráfico por stem comparando U-Net vs Open-Unmix."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    models = ["U-Net ResNet34", "Open-Unmix (Bi-LSTM)"]
    losses = [unet_loss, umx_loss]
    colors = ["#e9c46a", "#8d99ae"]  # Amarillo mostaza y Gris

    bars = ax.bar(models, losses, color=colors, width=0.45, edgecolor="#30363d", linewidth=1.5)
    
    # Ajustar límite Y proporcionalmente al valor máximo
    max_val = max([l for l in losses if l is not None] or [5.0])
    ax.set_ylim(0, max_val * 1.25)
    ax.set_ylabel("Pérdida en Test (L1 + MR-STFT)", fontsize=11, color="#c9d1d9")
    ax.set_title(f"Validación Test Set - Stem: {stem.upper()}", fontsize=13, fontweight="bold", color="#f0f6fc", pad=15)

    # Anotar valores sobre las barras
    for bar, loss, col in zip(bars, losses, colors):
        if loss is not None:
            ax.text(
                bar.get_x() + bar.get_width()/2.0, 
                bar.get_height() + (max_val * 0.03), 
                f"{loss:.4f}", 
                ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color=col
            )

    ax.grid(True, linestyle="--", alpha=0.2, color="#8b949e", axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()

    filename = os.path.join(out_dir, f"val_comparison_{stem}.png")
    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"    [★] Gráfico comparativo generado en: {filename}")

# ==========================================
# 3. EVALUACIÓN EN SUBSET DE TEST
# ==========================================
def evaluate_and_plot_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stems = ['vocals', 'drums', 'bass', 'other']
    segment_duration = 6.0
    batch_size = 4
    
    out_dir = "graphs"
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('dark_background')

    l1_criterion = nn.L1Loss()
    mr_stft_criterion = MultiResolutionSTFTLoss().to(device)

    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = torch.hann_window(win_length).to(device)

    print("\n=======================================================")
    print(f"=== EVALUACIÓN TEST SET - DISPOSITIVO: {device} ===")
    print("=======================================================\n")

    for stem in stems:
        print(f"\n---> Evaluando Stem: [{stem.upper()}]")
        
        test_dataset = MUSDBDataset(
            target_stem=stem, 
            segment_duration=segment_duration, 
            root='./data/musdb18hq', 
            subset='test',
            remix=False
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # 1. EVALUAR U-NET
        unet_loss = None
        unet_ckpt = os.path.join("checkpoints", "unet", f"unet_{stem}.pth")
        if os.path.exists(unet_ckpt):
            model_unet = get_unet_model().to(device)
            model_unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
            model_unet.eval()

            running_loss = 0.0
            with torch.no_grad():
                for mix_spec, target_spec, mix_audio, target_audio in test_loader:
                    mix_spec, target_spec = mix_spec.to(device), target_spec.to(device)
                    mix_audio, target_audio = mix_audio.to(device), target_audio.to(device)

                    pred_spec = model_unet(mix_spec)
                    loss_l1 = l1_criterion(pred_spec, target_spec)

                    mix_stft = torch.stft(
                        mix_audio.view(-1, mix_audio.shape[-1]), 
                        n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                        window=window, return_complex=True
                    )
                    
                    batch_size_curr = mix_spec.shape[0]
                    pred_spec_flat = pred_spec.view(-1, pred_spec.shape[2], pred_spec.shape[3])
                    pred_stft_complex = pred_spec_flat * torch.exp(1j * torch.angle(mix_stft))
                    
                    pred_audio_flat = torch.istft(
                        pred_stft_complex, n_fft=n_fft, hop_length=hop_length, 
                        win_length=win_length, window=window, length=mix_audio.shape[-1]
                    )
                    pred_audio = pred_audio_flat.view(batch_size_curr, 2, -1)

                    loss_mr_stft = mr_stft_criterion(pred_audio, target_audio)
                    loss = 2.0 * loss_l1 + 0.5 * loss_mr_stft
                    running_loss += loss.item()

            unet_loss = running_loss / len(test_loader)
            print(f"    [✔] U-Net ResNet34 Test Loss: {unet_loss:.4f}")

        # 2. EVALUAR OPEN-UNMIX
        umx_loss = None
        umx_ckpt = os.path.join("checkpoints", "openunmix", f"umx_{stem}.pth")
        if os.path.exists(umx_ckpt):
            model_umx = get_openunmix_model(n_fft=n_fft).to(device)
            model_umx.load_state_dict(torch.load(umx_ckpt, map_location=device))
            model_umx.eval()

            running_loss = 0.0
            with torch.no_grad():
                for mix_spec, target_spec, mix_audio, target_audio in test_loader:
                    mix_spec, target_spec = mix_spec.to(device), target_spec.to(device)
                    mix_audio, target_audio = mix_audio.to(device), target_audio.to(device)

                    pred_spec = model_umx(mix_spec)
                    loss_l1 = l1_criterion(pred_spec, target_spec)

                    mix_stft = torch.stft(
                        mix_audio.view(-1, mix_audio.shape[-1]), 
                        n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                        window=window, return_complex=True
                    )
                    
                    batch_size_curr = mix_spec.shape[0]
                    pred_spec_flat = pred_spec.view(-1, pred_spec.shape[2], pred_spec.shape[3])
                    pred_stft_complex = pred_spec_flat * torch.exp(1j * torch.angle(mix_stft))
                    
                    pred_audio_flat = torch.istft(
                        pred_stft_complex, n_fft=n_fft, hop_length=hop_length, 
                        win_length=win_length, window=window, length=mix_audio.shape[-1]
                    )
                    pred_audio = pred_audio_flat.view(batch_size_curr, 2, -1)

                    loss_mr_stft = mr_stft_criterion(pred_audio, target_audio)
                    loss = 2.0 * loss_l1 + 0.5 * loss_mr_stft
                    running_loss += loss.item()

            umx_loss = running_loss / len(test_loader)
            print(f"    [✔] Open-Unmix (Bi-LSTM) Test Loss: {umx_loss:.4f}")

        # Guardar gráfico comparativo de este stem
        if unet_loss is not None or umx_loss is not None:
            save_comparison_chart(stem, unet_loss, umx_loss, out_dir)

if __name__ == "__main__":
    evaluate_and_plot_test()