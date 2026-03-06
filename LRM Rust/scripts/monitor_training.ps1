# Training Log Monitoring Script
# Implements the Training Log Monitoring Plan - checks outputs, loss, and LR values.
# Run from project root: .\LRM Rust\scripts\monitor_training.ps1

$modelDir = "LRM Rust\models\vsrm_grid"
$experimentLog = Join-Path $modelDir "experiment.log"
$lossLog = Join-Path $modelDir "train\epoch-1\Loss.log"
$lrLog = Join-Path $modelDir "train\epoch-1\Learning_Rate.log"

Write-Host "=== VSRM Training Log Monitor ===" -ForegroundColor Cyan
Write-Host ""

# 1. Batch count
if (Test-Path $lossLog) {
    $lossLines = (Get-Content $lossLog).Count
    Write-Host "Batch count (epoch 1): $lossLines" -ForegroundColor Yellow
    if ($lossLines -lt 100) {
        Write-Host "  WARNING: Expected ~2500+ batches. Pipeline may have issues." -ForegroundColor Red
    } elseif ($lossLines -ge 2400) {
        Write-Host "  OK: Full epoch likely complete." -ForegroundColor Green
    } else {
        Write-Host "  In progress (~$([math]::Round($lossLines/2500*100, 1))% of expected epoch)." -ForegroundColor Gray
    }
} else {
    Write-Host "Loss.log not found at $lossLog" -ForegroundColor Red
}
Write-Host ""

# 2. Last 5 loss values
if (Test-Path $lossLog) {
    Write-Host "Last 5 loss values:" -ForegroundColor Yellow
    Get-Content $lossLog -Tail 5 | ForEach-Object { Write-Host "  $_" }
} 
Write-Host ""

# 3. Last 3 LR values
if (Test-Path $lrLog) {
    Write-Host "Last 3 LR values:" -ForegroundColor Yellow
    Get-Content $lrLog -Tail 3 | ForEach-Object { Write-Host "  $_" }
    $lastLr = (Get-Content $lrLog -Tail 1) -replace ",.*", ""
    $targetLr = 0.0003
    $pct = [double]$lastLr / $targetLr * 100
    Write-Host "  (Current LR is $([math]::Round($pct, 1))% of target 3e-4)" -ForegroundColor Gray
} else {
    Write-Host "Learning_Rate.log not found at $lrLog" -ForegroundColor Red
}
Write-Host ""

# 4. Recent predictions (last 3 samples)
if (Test-Path $experimentLog) {
    Write-Host "Recent predictions (last 3 samples):" -ForegroundColor Yellow
    Select-String -Path $experimentLog -Pattern "TRAIN SAMPLE|Target:|Preds" | Select-Object -Last 9 | ForEach-Object { Write-Host "  $($_.Line)" }
} else {
    Write-Host "experiment.log not found at $experimentLog" -ForegroundColor Red
}
Write-Host ""
Write-Host "=== End of monitor ===" -ForegroundColor Cyan
