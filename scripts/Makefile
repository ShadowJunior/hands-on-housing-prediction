# Makefile for hands-on ML project

.PHONY: env clean run-notebook

# Create environment using conda or venv+pip
env:
	bash setup_env.sh

# Remove Python virtual environment and conda env (if present)
clean:
	rm -rf .venv
	conda env remove -n handson-ml || true

# Launch JupyterLab
run-notebook:
	jupyter lab

# Install requirements with pip (if not using conda)
pip-install:
	pip install -r requirements.txt
