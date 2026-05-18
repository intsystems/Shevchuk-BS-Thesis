#!/bin/bash
# Останавливать скрипт при любой ошибке
set -e

rm -rf NeuroSTORM

git clone --recursive https://github.com/CUHK-AIM-Group/NeuroSTORM.git

wget -O neurostorm.ckpt "https://huggingface.co/zxcvb20001/NeuroSTORM/resolve/main/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt?download=true"

echo "=== 1. Очистка скрытого мусора и старого venv ==="
# Удаляем сломанный venv
rm -rf .venv
# КРИТИЧЕСКИЙ ШАГ: вычищаем пользовательскую директорию, откуда Python тайно импортировал старые либы
rm -rf ~/.local/lib/python3.10/site-packages/*

echo "=== 2. Создание чистого venv ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== 3. Обновление pip ==="
pip install --upgrade pip
echo "=== 4. Установка монолитного стека PyTorch (Согласованного с CUDA 12.9) ==="
# Явно указываем индекс для cu124/cu128, чтобы соответствовать системному nvcc 12.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== 5. Установка библиотек для диссертации ==="
pip install -r requirements.txt

echo "=== 5.5. Установка зависимостей сборки для Mamba ==="
# Сборка mamba_ssm требует предварительного наличия этих пакетов в окружении
pip install packaging wheel einops
# Ninja ускоряет и стабилизирует параллельную компиляцию C++ расширений
pip install ninja

echo "=== 5.6. Компиляция и установка СUDA-ядер Mamba ==="
# Устанавливаем без изоляции сборки, чтобы компилятор видел установленный torch и ninja
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

gcloud storage cp gs://fmrieeg-thesis-iowa/dataset.h5 .

gcloud storage cp gs://fmrieeg-thesis-iowa/eeg_channels.json .

echo "=== 6. Финальная проверка путей ==="
python3 -c "
import torch, torchvision, torchaudio, lightning
print('\n🎉  УСПЕХ: Окружение собрано без конфликтов C++!')
print('Torch path:', torch.__file__, f'({torch.__version__})')
print('Audio path:', torchaudio.__file__)