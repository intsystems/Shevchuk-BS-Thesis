#!/bin/bash
set -e

echo "=== 1. Клонирование репозитория и весов ==="
rm -rf NeuroSTORM
git clone --recursive https://github.com/CUHK-AIM-Group/NeuroSTORM.git
wget -O neurostorm.ckpt "https://huggingface.co/zxcvb20001/NeuroSTORM/resolve/main/pretraining/pt_neurostorm_mae_ratio0.5.ckpt?download=true"

echo "=== 2. Создание venv с наследованием ЗАВОДСКОГО PyTorch ==="
# Мы забираем идеальный, протестированный Google/NVIDIA торч из системы
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== 3. Установка пакетов диплома ==="
# Ставим только твои библиотеки, pip не будет трогать системный torch
pip install -r requirements.txt

echo "=== 4. Установка официальных колес Mamba-2 ==="
# Просто скачиваем готовые бинарники, они встанут за 3 секунды без компиляции
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0/causal_conv1d-1.5.0+cu124torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install causal_conv1d-1.5.0+cu124torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl --no-deps --force-reinstall

wget https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu124torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install mamba_ssm-2.3.0+cu124torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl --no-deps --force-reinstall

rm *.whl

echo "=== 5. Загрузка датасета ==="
gcloud storage cp gs://fmrieeg-thesis-iowa/dataset.h5 .
gcloud storage cp gs://fmrieeg-thesis-iowa/eeg_channels.json .

echo "=== 6. Финальная проверка ==="
python3 -c '
import torch
import causal_conv1d
import mamba_ssm
import lightning
print("\n🎉 🎉 🎉 ИДЕАЛЬНО: На чистом образе всё завелось из коробки!")
print("Версия Torch:", torch.__version__)
print("Доступность GPU:", torch.cuda.is_available())
'