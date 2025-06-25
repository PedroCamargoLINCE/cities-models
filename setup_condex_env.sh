#!/bin/bash

# setup_condex_env.sh
# Script to set up the Conda environment for the project.

# Environment name
ENV_NAME="codex_env"
PYTHON_VERSION="3.10"

# Path to the requirements file
REQ_FILE="requirements.txt"

echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating Conda environment..."
source activate $ENV_NAME || conda activate $ENV_NAME

echo "Updating pip and installation tools..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies from $REQ_FILE..."
pip install -r $REQ_FILE

echo "Installation complete."
echo "To activate this environment later, use:"
echo "conda activate $ENV_NAME"
