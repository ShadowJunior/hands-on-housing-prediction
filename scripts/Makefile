
# make env — Run the environment setup script (conda or venv+pip)
# make run-notebook — Launch JupyterLab
# make clean — Remove local venv and conda env
# make pip-install — Install requirements with pip
# You can now use simple make commands to automate setup and running your notebook.


# Makefile for hands-on housing prediction project

.PHONY: env clean run-notebook

# Create environment using conda or venv+pip
env:
	bash setup_env.sh

# Remove Python virtual environment and conda env (if present)
clean:
	rm -rf .venv
	conda env remove -n housing-prediction || true

# Launch JupyterLab
run-notebook:
	jupyter lab

# Install requirements with pip (if not using conda)
pip-install:
	pip install -r requirements.txt
