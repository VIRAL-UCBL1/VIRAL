# Welcome to VIRAL Installation Guide

## Requirements

- Python 3.11
- [Ollama](https://ollama.com/).
- [PyTorch](https://pytorch.org/).
- [CUDA](https://developer.nvidia.com/cuda)
- [Git](https://git-scm.com/).
- [Conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
git clone https://github.com/VIRAL-UCBL1/VIRAL.git
cd VIRAL
```


## Installation

**Before** installing the dependencies, create a **virtual environment** with **conda** or **venv**.

- Python requirements:

```bash
pip install -r requirements.txt
```

- [Ollama](https://ollama.com/download):

```bash
ollama run qwen2.5-coder
```

## Usage

```bash
cd src
python main.py
```

You can see help message by running:

```bash
python main.py -h
```
