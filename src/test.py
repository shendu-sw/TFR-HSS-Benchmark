"""
Runs a model on a single node across multiple gpus.
"""
import os
from pathlib import Path

import torch
from torch.backends import cudnn
import configargparse
import numpy as np
import pytorch_lightning as pl

from src.DeepRegression import Model


def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = Model(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    if hparams.gpu == 0:
        hparams.gpu = 0
    else:
        hparams.gpu = [hparams.gpu-1]
    trainer = pl.Trainer(
        gpus=hparams.gpu,
        precision=16 if hparams.use_16bit else 32,
        # limit_test_batches=0.05
    )

    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    model_path = list(Path(model_path).glob("*.ckpt"))[0]
    test_model = model.load_from_checkpoint(checkpoint_path=model_path, hparams=hparams)

    # ------------------------
    # 3 START PREDICTING
    # ------------------------
    print(hparams)
    print()

    trainer.test(model=test_model)

