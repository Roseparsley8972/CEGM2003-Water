# CEGM2003-Water

## Installation Instructions

### Prerequisites
- Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- NVIDIA GPU with CUDA support (required for tensorflow-gpu)
- CUDA Toolkit and cuDNN compatible with TensorFlow 2.10.1

### Create Environment
1. Clone this repository:
```cmd
git clone https://github.com/roseparsley8972/CEGM2003-Water.git
cd CEGM2003-Water
```

2. Creating and activating the conda environment:
```cmd
conda env create -f environment.yaml
conda activate GPU_compatible
```
