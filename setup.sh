#!/bin/bash
# Останавливать скрипт при любой ошибке
set -e

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

echo "=== 4. Установка монолитного стека PyTorch ==="
# Ставим строго вместе, чтобы pip зафиксировал совместимые версии внутри venv
pip install torch torchvision torchaudio

echo "=== 5. Установка библиотек для диссертации ==="
pip install -r requirements.txt

gcloud storage cp gs://fmrieeg-thesis-iowa/dataset.h5 .

gcloud storage cp gs://fmrieeg-thesis-iowa/eeg_channels.json .

echo "=== 6. Финальная проверка путей ==="
python3 -c "
import torch, torchvision, torchaudio, lightning
print('\n🎉 УСПЕХ: Окружение собрано без конфликтов C++!')
print('Torch path:', torch.__file__, f'({torch.__version__})')
print('Audio path:', torchaudio.__file__)
"