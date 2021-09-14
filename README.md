# Temperature field reconstruction implementation package
## Introduction
This project provides the implementation of the paper "A Machine Learning Modelling Benchmark for
Temperature Field Reconstruction of Heat-Source Systems". [[paper](https://arxiv.org/abs/2108.08298)] [[data generator](https://github.com/shendu-sw/recon-data-generator)]

## Requirements

* Software
    * python >= 3.6
    * cuda (only GPU is required)
    * pytorch
* Hardware
    * GPU with at least 16GB (recommended)
    * CPU

## Environment construction

1. Install required packages followed `requirements.txt`.

```python
pip install -r requirements.txt
```

2. Install `torch-cluster`, `torch-scatter`, `torch-sparse` package (matching the version of `torch`, `cuda`)

   * Automatic installation [[install instruction](https://github.com/rusty1s/pytorch_geometric#pip-wheels)]

     ```
     pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${version}+${CUDA}.html
     pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${version}+${CUDA}.html
     pip install torch-geometric
     pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${version}+${CUDA}.html
     pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${version}+${CUDA}.html
     ```

     `$version` describes the  `torch` version and should be replaced by `1.4.0`,`1.5.0`,`1.6.0`,`1.7.0`, `1.8.0`,`1.9.0`,`1.7.1`,`1.8.1`.

     `$CUDA` should be replaced by `cpu`, `cu101`, `cu102`, `cu111`, `cu92`.

   - Manual installation [[download](https://pytorch-geometric.com/whl)]

## Running
> All the methods for TFR-HSS task can be accessed by running `main.py` file.
>
> All the parameters are defined in two forms, namely the `yaml` file (default `config\config.yml`) and `command line` parameters. Priority: `yaml` < `command line` 

### Image-based and Vector-based methods

> The image-based and vector-based methods are following the same command.

- Training

  ```
  python main.py -m train
  ```

  or

  ```
  python main.py --mode=train
  ```

- Test

  ```
  python main.py -m test --test_check_num=21
  ```

  or

  ```
  python main.py --mode=test --test_check_num=21
  ```

  or

  ```
  python main.py -m=test -v=21
  ```

  where variable `test_check_num` is the number of the saved model for test.

- Prediction visualization

  ```
  python main.py -m plot --test_check_num=21
  ```

  or

  ```
  python main.py --mode=plot --test_check_num=21
  ```

  or

  ```
  python main.py -m=test -v=21
  ```

  where variable `test_check_num` is the number of the saved model for plotting.

### Point-based methods

> Only testing is permitted for point-based methods. 

- Testing
  ```
  python main.py
  ```


* Testing with prediction visualization

  ```
  python main.py --plot
  ```

## Project architecture

- `config`: the configuration file
  - `config.yml` describes configurations
    - `model_name`: model for reconstruction
    - `backbone`: backbone network, used only for deep surrogate models
    - `data_root`: root path of data
    - `train_list`: train samples
    - `test_list`: test samples
    - others
- `samples`: examples
- `outputs`: the output results by `test` and `plot` module. The test results is saved at `outputs/*.csv` and the plotting figures is saved at `outputs/predict_plot/`.
- `src`: including surrogate model, training and testing files.
  - `test.py`: testing files.
  - `train.py`: training files.
  - `plot.py`: prediction visualization files.
  - `point.py`: Model and testing files for point-based methods.
  - `DeepRegression.py`: Model configurations for image-based and vector-based methods.
  - `data`: data preprocessing and data loading files.
  - `models`: interpolation and machine learning models for the TFR-HSS task.
  - `utils`: useful tool function files.

* `docker`: start with docker.
* `lightning_logs`: saved models.

## One tiny example

One tiny example for training and testing can be accessed based on the following instruction.

- Some training and testing data are available at `samples/data`.
- Based on the original configuration file, run `python main.py` directly for a quick experience of this tiny example.

## Citing this work

If you find this work helpful for your research, please consider citing:

```
@article{gong2021,
    Author = {Xiaoqian Chen and Zhiqiang Gong and Xiaoyu Zhao and Weien Zhou and Wen Yao},
    Title = {A Machine Learning Modelling Benchmark for Temperature Field Reconstruction of Heat-Source Systems},
    Journal = {arXiv preprint arXiv:2108.08298},
    Year = {2021}
}
```