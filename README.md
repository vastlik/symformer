# SymFormer: End-to-end symbolic regression using transformer-based architecture

This repository contains the official implementation of SymFormer. It is a symbolic regression method that uses a
transformer model to generate a symbolic representation of a function based on the function's output.

[Paper](https://arxiv.org/pdf/2205.15764)&nbsp;&nbsp;&nbsp;
[Web](https://vastlik.github.io/symformer/)&nbsp;&nbsp;&nbsp;
[Demo](https://colab.research.google.com/github/vastlik/symformer/blob/main/notebooks/symformer-playground.ipynb)

<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg?style=for-the-badge)](https://colab.research.google.com/github/vastlik/symformer/blob/main/notebooks/symformer-playground.ipynb)
![Python Versions](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)

<br>

## Getting started

Start by creating a Python 3.9 venv. From the activated environment, you can run the following command in the repository
root:

```
pip install -r requirements.txt
```

## Getting datasets

To generate a one-dimensional dataset (used to train the univariate model) run the following commands:

```
python -m symformer generate-dataset \
    --output-dir general/train \
    --dataset-size 130000000 \
    --n-processes 128 \
    --seed 1234
python -m symformer generate-dataset \
    --output-dir general/valid \
    --dataset-size 10000 \
    --n-processes 128 \
    --seed 5678
```

To generate a two-dimensional dataset (used to train the bivariate model) run the following commands:

```
python -m symformer generate-dataset \
    --output-dir general/train \
    --dataset-size 100000000 \
    --n-processes 128 \
    --seed 1234 \
    --num-variables 2
python -m symformer generate-dataset \
    --output-dir general/valid \
    --dataset-size 10000 \
    --n-processes 128 \
    --seed 5678 \
    --num-variables 2
```

For further hyperparameters see `python -m symformer generate-dataset --help`.

## Running the inference

You can run your model by selecting your own trained model for `--model` param or specifying one of the
`symformer-univariate` or `symformer-bivariate` which will download the model from the repository.

### Single equation

To run a single equation:

```
python -m symformer predict --model symformer-univariate 'sin(x**2)'
```

Output:

```
Function: sin(((x)^2))
R2: 1.0
Relative error: 5.582490629923639e-16
```

You can also change the model to your own model.

### Benchmark functions

To run the benchmark use command bellow:

```
python -m symformer evaluate-benchmark --univariate-model symformer-univariate --bivariate-model symformer-bivariate
```

### Evaluation on dataset

To run the evaluation on dataset run the following:

```
python -m symformer evaluate --model symformer-univariate --test-dataset-path path/to/datast
```

### Running equation prediction inside code

You can also run the code from the python using the `Runner` class. Example of such code is in
`notebooks/symformer-playground.ipynb`.

```python
from symformer.model.runner import Runner

runner = Runner.from_checkpoint('symformer-univariate')
prediction, r2, relative_error = runner.predict('sin(x)')
print(prediction, r2, relative_error)
```

Output:

```
sin(x) 1.0 0.0
```

or for bivariate functions:

```python
from symformer.model.runner import Runner

runner = Runner.from_checkpoint('symformer-bivariate')
prediction, r2, relative_error = runner.predict('sin(x+y)')
print(prediction, r2, relative_error)
```

Output:

```
sin(x+y) 1.0 0.0
```

## Training a model from scratch

To train a model run the following:

```
python -m symformer train \
    --config configs/{config name}.json \
    --dataset-path /path/to/train/dataset/ \
    --dataset-valid-path /path/to/valid/dataset/
```

where `{config name}` is is one of the files contained in the `configs` directory.

## Citation

If you found our work useful, please use the following citation:

```
@article{vastl2022symformer,
  title={SymFormer: End-to-end symbolic regression using transformer-based architecture},
  author={Vastl, Martin and Kulh{\'a}nek, Jon{\'a}{\v{s}} and Kubal{\'i}k, Ji{\v{r}}{\'i} and Derner, Erik and Babu{\v{s}}ka, Robert},
  journal={arXiv preprint arXiv:2205.15764},
  year={2022},
}
```
