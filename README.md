## About

A convenient module for working with quantization of models.

The project is based on PyTorch's
[FX/Eager](https://pytorch.org/docs/stable/quantization.html) technology.

## Install

```bash
git clone https://github.com/vd-kuznetsov/qtools.git

cd qtools
conda create --name qtools python==3.8
conda activate qtools

poetry install
pre-commit install
pre-commit run -a
```

## Usage

Recommended actions:

1. Launch the MLflow server, check if your address matches the one specified in
   the config

2. `python commands.py`

The second point can be run separately:

- `python commands.py mode=train` - model training

  - Downloads a dataset using DVC from GDrive

- `python commands.py mode=infer` - quantizes the trained model with a single
  line of code

  - By default, the weights of the model are used after training
