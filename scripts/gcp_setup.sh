#!/bin/bash
# GCP Setup Script for LIBERO Evaluation
# Run this on your GCP instance

set -e

echo "=== PSA-SNN LIBERO Evaluation Setup ==="

# Create conda environment
conda create -n psa-libero python=3.10 -y
conda activate psa-libero

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install LIBERO dependencies
pip install robosuite
pip install libero

# Clone PSA-SNN
git clone https://github.com/yubo-ruan/psa-snn.git
cd psa-snn
pip install -r requirements.txt

# Install additional dependencies for LIBERO
pip install h5py imageio wandb

echo "=== Setup Complete ==="
echo "To run evaluation: python run_libero_eval.py"
