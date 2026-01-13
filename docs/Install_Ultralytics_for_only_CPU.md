# 🧠 Install Ultralytics library for CPU-only usage on Linux:

To install the Ultralytics library for CPU-only usage on Linux, you must first **ensure that only the CPU version of PyTorch is installed**, *as the standard installation typically defaults to the GPU-enabled version if CUDA is available.*

Here are the steps using pip to force a CPU-only installation:

## Prerequesites:
It is highly recommended to use a virtual environment to avoid conflicts with your system's Python packages.


```bash
# Create a virtual environment (named 'ultralytics_cpu_env')
python3 -m venv ultralytics_cpu_env

# Activate the environment
source ultralytics_cpu_env/bin/activate
```

## Installation Steps

1. **Install the CPU-only version of PyTorch**: This step is crucial for preventing the installation of CUDA dependencies. A common command for the CPU-only version is:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

This command directs `pip3` to the PyTorch CPU-only package index.

2. **Install the Ultralytics library**: Once the CPU-only PyTorch is successfully installed, you can install the Ultralytics package.

```bash 
pip3 install ultralytics
```

`pip` will recognize that a compatible version of PyTorch is already installed and will skip reinstalling it with potential GPU dependencies. 

## Verification

After installation, you can verify that Ultralytics is installed and using the CPU by running a simple Python script in your activated environment: 

```bash 
python -c "import torch; print(f'Torch installed: {torch.__version__}'); print(f'Cuda available: {torch.cuda.is_available()}')"
```

The output for `Cuda available` should be `False`. 

You can check the details of YOLO as follows:

```bash 
yolo check
```

This will show you all details of installed Ultralytics

## Save the all work inside the container:

### First deactivate the Virtual Environment:

```bash
deactivate
```
### Save the packages you have installed:

```bash
# exit from the container (Note that : exit NOT stop the container)
exit
```
```bash
docker commit parol6_dev parol6-ultimate:latest
```
This will save any libraries or packages you have installed inside the container.  **You can now stop the container in safe way.**

