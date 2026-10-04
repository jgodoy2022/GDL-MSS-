Write-Host "===============================================================" -ForegroundColor Green
Write-Host " INICIANDO ENTRENAMIENTO COMPLETO NOCTURNO (U-Net + Open-Unmix)" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green

$stems = @("vocals", "drums", "bass", "other")

# 1. Ejecutar U-Net ResNet34
foreach ($stem in $stems) {
    Write-Host "`n[1/2] Entrenando U-Net para: [$stem]..." -ForegroundColor Yellow
    python -u train_unet.py $stem
}

# 2. Ejecutar Open-Unmix Bi-LSTM
foreach ($stem in $stems) {
    Write-Host "`n[2/2] Entrenando Open-Unmix para: [$stem]..." -ForegroundColor Cyan
    python -u train_umx.py $stem
}

Write-Host "`n===============================================================" -ForegroundColor Green
Write-Host " ¡TODOS LOS ENTRENAMIENTOS HAN FINALIZADO!" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green