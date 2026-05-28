#!/bin/bash
set -e

echo "=== 1. Клонирование репозитория и весов ==="
rm -rf NeuroSTORM
git clone --recursive https://github.com/CUHK-AIM-Group/NeuroSTORM.git
wget -O neurostorm.ckpt "https://huggingface.co/zxcvb20001/NeuroSTORM/resolve/main/pretraining/pt_neurostorm_mae_5ds.ckpt?download=true"

echo "=== 2. Создание venv с наследованием ЗАВОДСКОГО PyTorch ==="
# Мы забираем идеальный, протестированный Google/NVIDIA торч из системы
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== 3. Установка torch cu128 (как в Colab) ==="
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== 4. Установка пакетов диплома ==="
pip install -r requirements.txt

echo "=== 5. Установка официальных колес Mamba-2 для torch2.10 ==="
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-deps --force-reinstall

wget https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-deps --force-reinstall

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