import os
import json
import matplotlib.pyplot as plt

def plot_training_charts():
    log_dir = "logs"
    out_dir = "graphs"
    os.makedirs(out_dir, exist_ok=True)
    
    stems = ["vocals", "drums", "bass", "other"]
    
    # Aplicar estilo oscuro
    plt.style.use('dark_background')
    
    for stem in stems:
        unet_path = os.path.join(log_dir, f"unet_history_{stem}.json")
        umx_path = os.path.join(log_dir, f"umx_history_{stem}.json")
        
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        
        # Cargar datos U-Net (Amarillo Mostaza)
        if os.path.exists(unet_path):
            with open(unet_path, "r") as f:
                data_unet = json.load(f)
            epochs_unet = [d["epoch"] for d in data_unet]
            losses_unet = [d["loss"] for d in data_unet]
            ax.plot(epochs_unet, losses_unet, label="U-Net ResNet34", color="#e9c46a", linewidth=2.5)
            
        # Cargar datos Open-Unmix (Gris)
        if os.path.exists(umx_path):
            with open(umx_path, "r") as f:
                data_umx = json.load(f)
            epochs_umx = [d["epoch"] for d in data_umx]
            losses_umx = [d["loss"] for d in data_umx]
            ax.plot(epochs_umx, losses_umx, label="Open-Unmix (Bi-LSTM)", color="#8d99ae", linewidth=2.5)
            
        # Títulos y etiquetas
        ax.set_title(f"Comparativa Loss - Stem: {stem.upper()}", fontsize=14, fontweight="bold", pad=15, color="#f0f6fc")
        ax.set_xlabel("Época", fontsize=12, color="#c9d1d9")
        ax.set_ylabel("Loss (L1 + MR-STFT)", fontsize=12, color="#c9d1d9")
        
        # Ajustar límites de los ejes (Eje Y limitado entre 0 y 10)
        ax.set_ylim(0, 10)
        
        # Rejilla
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.3, color="#8b949e")
        
        # Leyenda ubicada fijamente en la esquina superior derecha
        ax.legend(loc='upper right', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
        
        plt.tight_layout()
        
        chart_file = os.path.join(out_dir, f"comparison_{stem}.png")
        plt.savefig(chart_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"[✔] Gráfico generado en: {chart_file}")

if __name__ == "__main__":
    plot_training_charts()