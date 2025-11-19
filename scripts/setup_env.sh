#!/bin/bash
# Setup script for hands-on housing prediction environment
set -e

# Create conda environment if conda is available
if command -v conda &> /dev/null; then
  echo "Creating conda environment from environment.yml..."
  conda env create -f environment.yml || conda env update -f environment.yml
  echo "Activate with: conda activate housing-prediction"
else
  echo "Conda not found. Using venv and pip."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  echo "Activate with: source .venv/bin/activate"
fi

# Final message
echo "Environment setup complete. Launch JupyterLab with: jupyter lab"
