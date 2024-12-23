
#!/bin/bash

# Set path to root and novic repository

export LOC=/home/user/

export NOVIC="$LOC/novic"


# Create conda env for novic
export NENV=novic

conda create -n "$NENV" python=3.10
conda activate "$NENV"
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge datasets filelock 'huggingface_hub<1' packaging pyyaml regex requests safetensors tqdm certifi openssl ca-certificates
conda install -c conda-forge transformers==4.38.2  # <-- Other versions most likely work (explicit version avoids conda dependency mismanagement wrt optimum, but takes a LONG time anyway)
conda install -c conda-forge optimum==1.17.1  # <-- Other versions most likely work (explicit version avoids conda dependency mismanagement wrt transformers, but takes a LONG time anyway)
conda install -c conda-forge hydra-core accelerate ftfy timm sentencepiece wandb unidecode tabulate
pip install openai opencv-python
pip install open_clip_torch==2.23 git+https://github.com/openai/CLIP.git
pip check

# Loging to weights and biases with api key.
wandb login 6a656aae688674e7e3e14d8562d04794c393d00f


# Check if everything is ok

./train.py --help
