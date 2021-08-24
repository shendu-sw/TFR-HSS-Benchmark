# encoding: utf-8
"""
This function denotes the main function to train/test/plot
Usage:
    python main.py [FLAGS]

@author: gongzhiqiang
@contact: gongzhiqiang@alumni.sjtu.edu.cn

@version: 1.0
@file: main.py
@time: 2021-06-29

"""
from pathlib import Path
import configargparse

from src.DeepRegression import Model
from src import train, test, plot, point


def main():
    # default configuration file
    config_path = Path(__file__).absolute().parent / "config/config.yml"
    data_path = Path(__file__).absolute().parent / "config/data.yml"
    parser = configargparse.ArgParser(config_file_parser_class= configargparse.YAMLConfigFileParser, \
        default_config_files=[str(config_path), str(data_path)], description="Hyper-parameters.")
    
    # configuration file
    parser.add_argument("--config", is_config_file=True, default=False, help="config file path")
    
    # mode
    parser.add_argument("-m", "--mode", type=str, default="train", help="model: train or test or plot")

    # problem dimension
    parser.add_argument("--prob_dim", default=2, type=int, help="dimension of the problem")

    # args for plot in point-based methods
    parser.add_argument("--plot", action="store_true", help="use profiler")

    # args for training
    parser.add_argument("--gpu", type=int, default=0, help="which gpu: 0 for cpu, 1 for gpu 0, 2 for gpu 1, ...")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--lr", default="0.01", type=float)
    parser.add_argument("--resume_from_checkpoint", type=str, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=2, type=int, help="num_workers in DataLoader")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--use_16bit", type=bool, default=False, help="use 16bit precision")
    parser.add_argument("--profiler", action="store_true", help="use profiler")

    # args for validation
    parser.add_argument("--val_check_interval", type=float, default=1,
                        help="how often within one training epoch to check the validation set")

    # args for testing
    parser.add_argument("-v", "--test_check_num", default='0', type=str, help="checkpoint for test")
    parser.add_argument("--test_args", action="store_true", help="print args")

    # args from Model
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    PointModel = ["RSVR", "RBM", "RandomForest", "Polynomial", "MLPP", "Kriging", "KInterpolation", "GInterpolation"]
    dpoint = True if hparams.model_name in PointModel else False
    # running
    assert hparams.mode in ["train", "test", "plot"]
    if hparams.test_args:
        print(hparams)
    elif dpoint:
        print(f"Only testing is permitted for", hparams.model_name)
        point.main(hparams)
    else:
        getattr(eval(hparams.mode), "main")(hparams)


if __name__ == '__main__':
    main()
