#!/bin/bash

# ==============================================================================
# AUTOMATED SETUP: QWEN2.5-VL + VADER SENTIMENT PROJECT
# Environment: 2xT4 GPUs | CUDA 12.8
# ==============================================================================

ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

echo "--- Starting full installation for $ENV_NAME ---"

# 1. Initialize Conda for this script session
# This ensures 'conda activate' works within the bash process
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. Fresh Environment Creation
echo "[1/7] Creating Conda environment..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 3. Activation
echo "[2/7] Activating environment..."
conda activate $ENV_NAME

# 4. Specific PyTorch + CUDA 12.8 Installation
echo "[3/7] Installing PyTorch 2.11.0 with CUDA 12.8 support..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5. Bulk Installation from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "[4/7] Installing all packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found in current directory!"
    exit 1
fi

# 6. Safety Enforcer for Hugging Face Hub
# Re-asserting version < 1.0 to prevent 'transformers' from upgrading it
echo "[5/7] Enforcing huggingface-hub version constraint (<1.0)..."
pip install "huggingface-hub>=0.30.0,<1.0"

# echo "[6/7] Hugging Face CLI download..."
# pip install -U "huggingface_hub[cli]"

echo "[7/7]Model Download Started: Qwen2.5-VL 7B Instruct"
hf download Qwen/Qwen2.5-VL-7B-Instruct


# Final Status Check
echo "------------------------------------------------"
echo "INSTALLATION SUCCESSFUL"
echo "------------------------------------------------"
echo "Environment: $ENV_NAME"
echo "Python: $(python --version)"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF Hub Version: $(python -c 'import huggingface_hub; print(huggingface_hub.__version__)')"