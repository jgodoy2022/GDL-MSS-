import os
import torch
import soundfile as sf
import musdb
from src.models_umx import get_openunmix_model

def separate_umx_stem(target_stem='drums', track_index=2, use_wiener=False, power=1.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_str = f"WIENER (power={power})" if use_wiener else "DIRECTO (Sin filtro - Máxima nitidez)"
    print(f"--- Evaluando Open-Unmix para [{target_stem.upper()}] | Modo: {mode_str} ---")

    output_dir = os.path.join("samples", "umx_test")
    os.makedirs(output_dir, exist_ok=True)

    sample_rate = 44100
    duration = 15.0
    start_sec = 30.0

    # 1. Cargar pista de prueba
    mus = musdb.DB(root='./data/musdb18hq', is_wav=True, subsets='test')
    track = mus.tracks[track_index]
    track.chunk_start = start_sec
    track.chunk_duration = duration
    track_name = f"{track.artist}_-_{track.title}".replace(" ", "_")

    mix_audio = torch.tensor(track.audio.T, dtype=torch.float32)

    # Exportar mezcla de referencia y la pista original (Ground Truth)
    sf.write(os.path.join(output_dir, f"{track_name}_00_MIXTURE.wav"), mix_audio.T.numpy(), sample_rate)
    sf.write(os.path.join(output_dir, f"{track_name}_ORIGINAL_{target_stem}.wav"), track.targets[target_stem].audio, sample_rate)

    # 2. Configurar STFT
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
    mix_spec = torch.abs(mix_stft).unsqueeze(0).to(device) # (1, 2, F, T)

    # 3. Cargar modelo Open-Unmix
    checkpoint_path = f"umx_{target_stem}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: No se encontró '{checkpoint_path}'.")
        return

    model = get_openunmix_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    print(f"--> Procesando {target_stem} con Open-Unmix...")
    with torch.no_grad():
        pred_spec = model(mix_spec).squeeze(0).cpu() # (2, F, T)

    mix_spec_cpu = mix_spec.squeeze(0).cpu()

    # --- APLICACIÓN DE MÁSCARA ---
    if use_wiener:
        # Modo Filtro Wiener (Soft Masking)
        pred_other_spec = torch.clamp(mix_spec_cpu - pred_spec, min=1e-7)
        target_pow = torch.pow(pred_spec, power)
        other_pow = torch.pow(pred_other_spec, power)
        mask = target_pow / (target_pow + other_pow + 1e-7)
        final_stft_complex = (mix_spec_cpu * mask) * torch.exp(1j * torch.angle(mix_stft))
        tag = "UMX_WIENER"
    else:
        # Modo Directo (Hard Masking) - Preserva 100% el brillo y nitidez
        final_stft_complex = pred_spec * torch.exp(1j * torch.angle(mix_stft))
        tag = "UMX_DIRECT"

    estimated_audio = torch.istft(
        final_stft_complex, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window, 
        length=mix_audio.shape[-1]
    )

    out_file = os.path.join(output_dir, f"{track_name}_{tag}_{target_stem}.wav")
    sf.write(out_file, estimated_audio.T.numpy(), sample_rate)

    print(f"\n[✔] Pista generada con éxito: {out_file}")

if __name__ == "__main__":
    # use_wiener=False  --> Genera la versión original nítida
    # use_wiener=True   --> Aplica el filtro Wiener
    separate_umx_stem(target_stem='other', track_index=2, use_wiener=False)