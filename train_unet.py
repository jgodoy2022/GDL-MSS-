import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import MUSDBDataset
from src.models import get_unet_model
from src.losses import MultiResolutionSTFTLoss
from tqdm import tqdm

def train(target_stem='drums'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"--- Entrenando U-Net ResNet34 para STEM: [{target_stem.upper()}] en {torch.cuda.get_device_name(0)} ---")
    print(f"=======================================================\n")

    batch_size = 2
    epochs = 150
    learning_rate = 1e-3
    segment_duration = 8.0

    print(f"Cargando MUSDB18-HQ (Target: {target_stem} con Stem Remixing)...")
    train_dataset = MUSDBDataset(
        target_stem=target_stem, 
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

    model = get_unet_model().to(device)
    
    l1_criterion = nn.L1Loss()
    mr_stft_criterion = MultiResolutionSTFTLoss().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = torch.hann_window(win_length).to(device)

    checkpoint_name = f"unet_{target_stem}.pth"
    best_loss = float('inf')
    
    # Configuración de Logging
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"unet_history_{target_stem}.json")
    history = []
    
    print(f"Inicio de entrenamiento ({epochs} Épocas)...")
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"[{target_stem.upper()}] Época [{epoch+1}/{epochs}]", leave=True)

        for mix_spec, target_spec, mix_audio, target_audio in pbar:
            mix_spec = mix_spec.to(device, non_blocking=True)
            target_spec = target_spec.to(device, non_blocking=True)
            mix_audio = mix_audio.to(device, non_blocking=True)
            target_audio = target_audio.to(device, non_blocking=True)

            optimizer.zero_grad()

            mask = model(mix_spec)
            pred_spec = mask * mix_spec

            loss_l1 = l1_criterion(pred_spec, target_spec)

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
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.6f}"})

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        
        # Registrar métricas
        history.append({
            "epoch": epoch + 1,
            "loss": float(epoch_loss),
            "lr": float(current_lr),
            "time_sec": float(elapsed_time)
        })
        with open(log_file, "w") as f:
            json.dump(history, f, indent=4)
        
        print(f"--> Época [{epoch+1}/{epochs}] - Loss Promedio: {epoch_loss:.4f} - Tiempo: {elapsed_time:.2f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_name)
            print(f"    [★] ¡Nueva mejor Loss ({best_loss:.4f})! Checkpoint '{checkpoint_name}' actualizado.")

    print(f"\n¡Entrenamiento de {target_stem.upper()} completado! Mejor Loss: {best_loss:.4f}")

if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else 'drums'
    train(stem)