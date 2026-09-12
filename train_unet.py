import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import MUSDBDataset
from src.models import UNet2D
from tqdm import tqdm
import time

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Entrenando U-Net (Vocals Extended) en: {torch.cuda.get_device_name(0)} ---")

    # Hyperparámetros extendidos
    batch_size = 4
    epochs = 40          # Incrementamos a 40 épocas para buscar convergencia
    learning_rate = 1e-3
    segment_duration = 4.0

    print("Cargando MUSDB18-HQ (Train)...")
    train_dataset = MUSDBDataset(
        target_stem='vocals', 
        segment_duration=segment_duration, 
        root='./data/musdb18hq', 
        subset='train'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )

    model = UNet2D(in_channels=2, out_channels=2).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler para reducir suavemente el LR hacia las últimas épocas
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float('inf')
    print(f"Inicio de entrenamiento ({epochs} Épocas)...")
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Época [{epoch+1}/{epochs}]", leave=True)

        for mix_spec, target_spec, _ in pbar:
            mix_spec = mix_spec.to(device, non_blocking=True)
            target_spec = target_spec.to(device, non_blocking=True)

            optimizer.zero_grad()

            mask = model(mix_spec)
            pred_spec = mask * mix_spec

            loss = criterion(pred_spec, target_spec)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss L1': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.6f}"})

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        
        print(f"--> Época [{epoch+1}/{epochs}] - Loss Promedio: {epoch_loss:.4f} - Tiempo: {elapsed_time:.2f}s")

        # Guardar únicamente si es la mejor Loss hasta el momento
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "unet_vocals.pth")
            print(f"    [★] ¡Nueva mejor Loss ({best_loss:.4f})! Checkpoint 'unet_vocals.pth' actualizado.")

    print(f"\n¡Entrenamiento completado! Mejor Loss alcanzada: {best_loss:.4f}")

if __name__ == "__main__":
    train()