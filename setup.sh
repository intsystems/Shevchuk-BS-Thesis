#!/bin/bash
# Останавливать скрипт при любой ошибке
set -e

echo "=== 1. Клонирование репозитория и скачивание чекпоинта ==="
rm -rf NeuroSTORM
git clone --recursive https://github.com/CUHK-AIM-Group/NeuroSTORM.git

# Качаем веса NeuroSTORM
wget -O neurostorm.ckpt "https://huggingface.co/zxcvb20001/NeuroSTORM/resolve/main/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt?download=true"

echo "=== 2. Очистка старого venv и кэша пакетов ==="
rm -rf .venv
# Исправлено: чистим глобальный кэш под актуальный Python 3.12, чтобы старые бинарники не всплыли
rm -rf ~/.local/lib/python3.12/site-packages/*

echo "=== 3. Создание чистого venv и обновление pip ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "=== 4. Установка PyTorch 2.4.0 (Совместимого с колесами Mamba) ==="
# Берем cu124, так как под нее собраны стабильные бинарники Мамбы
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== 5. Установка основного стека из requirements.txt ==="
pip install -r requirements.txt

echo "=== 6. Быстрая установка готовых колес Mamba-2 (Без компиляции) ==="
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post2/causal_conv1d-1.5.3.post1+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install causal_conv1d-1.5.3.post1+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
rm causal_conv1d-*.whl

wget https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install mamba_ssm-2.3.0+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl --no-deps
rm mamba_ssm-*.whl

echo "=== 7. Загрузка данных диссертации из Google Cloud Storage ==="
gcloud storage cp gs://fmrieeg-thesis-iowa/dataset.h5 .
gcloud storage cp gs://fmrieeg-thesis-iowa/eeg_channels.json .

echo "=== 8. Финальная проверка окружения ==="
python3 -c '
import torch
import causal_conv1d
import mamba_ssm
import lightning
print("\n🎉 🎉 🎉 УСПЕХ: Все тяжелые CUDA-модули и Mamba-2 импортируются без конфликтов!")
print("Версия Torch:", torch.__version__)
print("Доступность GPU:", torch.cuda.is_available())
'