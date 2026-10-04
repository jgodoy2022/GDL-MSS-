import os
import torch
import soundfile as sf
import musdb
from src.models import get_unet_model

def separate_and_compare(track_index=2, custom_audio_path=None):
    """
    track_index: Índice de la canción en MUSDB18-HQ test set (0 a 49).
    custom_audio_path: Ruta a un archivo propio (si se usa, no hay stems originales de referencia).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Iniciando Separación 4-STEM con U-Net (Máscara Directa / Sin Filtro) en: {device} ---")

    output_dir = os.path.join("samples", "stems_separated")
    os.makedirs(output_dir, exist_ok=True)

    sample_rate = 44100
    duration = 15.0
    start_sec = 30.0

    has_ground_truth = False

    if custom_audio_path and os.path.exists(custom_audio_path):
        print(f"Cargando archivo personalizado: '{custom_audio_path}'...")
        audio_data, sr = sf.read(custom_audio_path)
        if sr != sample_rate:
            print(f"Advertencia: Muestra a {sr}Hz. Se recomienda 44100Hz.")
        mix_audio = torch.tensor(audio_data.T, dtype=torch.float32)
        track_name = os.path.splitext(os.path.basename(custom_audio_path))[0]
    else:
        mus = musdb.DB(root='./data/musdb18hq', is_wav=True, subsets='test')
        track = mus.tracks[track_index]
        track.chunk_start = start_sec
        track.chunk_duration = duration
        track_name = f"{track.artist}_-_{track.title}".replace(" ", "_")
        has_ground_truth = True
        print(f"Cargando canción de prueba #{track_index}: '{track.artist} - {track.title}' ({duration}s)...")
        mix_audio = torch.tensor(track.audio.T, dtype=torch.float32)

    # 1. Guardar la mezcla original completa
    sf.write(os.path.join(output_dir, f"{track_name}_00_MIXTURE.wav"), mix_audio.T.numpy(), sample_rate)

    # 2. Configurar STFT de la mezcla
    n_fft = 2048
    hop_length = 512
    win_length = 2048
    window = torch.hann_window(win_length)

    mix_stft = torch.stft(
        mix_audio, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window, 
        return_complex=True
    )
    mix_spec = torch.abs(mix_stft).unsqueeze(0).to(device)

    stems = ['vocals', 'drums', 'bass', 'other']

    # 3. Exportar las referencias originales (Ground Truth) de MUSDB18-HQ si existen
    if has_ground_truth:
        print("\n--> Exportando pistas de referencia originales (Ground Truth)...")
        for stem in stems:
            gt_audio = track.targets[stem].audio
            sf.write(os.path.join(output_dir, f"{track_name}_ORIGINAL_{stem}.wav"), gt_audio, sample_rate)
            print(f"    [GT] Guardado: '{track_name}_ORIGINAL_{stem}.wav'")

    # 4. Procesar y exportar la estimación de cada modelo U-Net (Sin filtro)
    print("\n--> Procesando separación directa con los modelos U-Net ResNet34...")
    for stem in stems:
        checkpoint_path = f"unet_{stem}.pth"
        
        if os.path.exists(checkpoint_path):
            model = get_unet_model().to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            model.eval()

            with torch.no_grad():
                mask = model(mix_spec)

            mask = mask.squeeze(0).cpu()

            # --- MÁSCARA DIRECTA (Hard Masking - Preserva brillo sin opacar) ---
            separated_stft = mask * mix_stft

            estimated_audio = torch.istft(
                separated_stft, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window, 
                length=mix_audio.shape[-1]
            )

            estimated_audio_np = estimated_audio.T.numpy()
            sf.write(os.path.join(output_dir, f"{track_name}_UNET_DIRECT_{stem}.wav"), estimated_audio_np, sample_rate)
            print(f"    [MODEL] Guardado: '{track_name}_UNET_DIRECT_{stem}.wav'")
        else:
            print(f"    [!] Checkpoint '{checkpoint_path}' no encontrado.")

    print(f"\n¡Proceso finalizado! Revisa la carpeta: {output_dir}")

if __name__ == "__main__":
    separate_and_compare(track_index=2)