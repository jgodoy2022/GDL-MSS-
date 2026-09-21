import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import MUSDBDataset
from src.models import get_unet_model
from src.losses import MultiResolutionSTFTLoss
from tqdm import tqdm
import time

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Entrenando U-Net Profesional (ResNet34) en: {torch.cuda.get_device_name(0)} ---")

    # Hiperparámetros
    batch_size = 2         # Proteger VRAM para muestras de 8s y modelo más profundo
    epochs = 100
    learning_rate = 1e-3
    segment_duration = 8.0

    print("Cargando MUSDB18-HQ (Train con Stem Remixing)...")
    train_dataset = MUSDBDataset(
        target_stem='vocals', 
        segment_duration=segment_duration, 
        root='./data/musdb18hq', 
        subset='train',
        remix=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )

    # Importamos el modelo U-Net profesional desde librería
    model = get_unet_model().to(device)
    
    l1_criterion = nn.L1Loss()
    mr_stft_criterion = MultiResolutionSTFTLoss().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = torch.hann_window(win_length).to(device)

    best_loss = float('inf')
    print(f"Inicio de entrenamiento ({epochs} Épocas)...")
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Época [{epoch+1}/{epochs}]", leave=True)

        for mix_spec, target_spec, mix_audio, target_audio in pbar:
            mix_spec = mix_spec.to(device, non_blocking=True)
            target_spec = target_spec.to(device, non_blocking=True)
            mix_audio = mix_audio.to(device, non_blocking=True)
            target_audio = target_audio.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Predicción de la máscara con U-Net ResNet34
            mask = model(mix_spec)
            pred_spec = mask * mix_spec

            loss_l1 = l1_criterion(pred_spec, target_spec)

            # Reconstrucción temporal
            mix_stft = torch.stft(
                mix_audio.view(-1, mix_audio.shape[-1]), 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                return_complex=True
            )
            
            batch_size_curr = mix_spec.shape[0]
            pred_spec_flat = pred_spec.view(-1, pred_spec.shape[2], pred_spec.shape[3])
            pred_stft_complex = pred_spec_flat * torch.exp(1j * torch.angle(mix_stft))
            
            pred_audio_flat = torch.istft(
                pred_stft_complex, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window,
                length=mix_audio.shape[-1]
            )
            
            pred_audio = pred_audio_flat.view(batch_size_curr, 2, -1)

            loss_mr_stft = mr_stft_criterion(pred_audio, target_audio)

            loss = 2.0 * loss_l1 + 0.5 * loss_mr_stft

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss Total': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.6f}"})

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        
        print(f"--> Época [{epoch+1}/{epochs}] - Loss Promedio: {epoch_loss:.4f} - Tiempo: {elapsed_time:.2f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "unet_vocals.pth")
            print(f"    [★] ¡Nueva mejor Loss ({best_loss:.4f})! Checkpoint 'unet_vocals.pth' actualizado.")

    print(f"\n¡Entrenamiento completado! Mejor Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()