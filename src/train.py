"""
Runs a model on a single node across multiple gpus.
"""
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
    #print(hparams.gpus)
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpu,
        precision=16 if hparams.use_16bit else 32,
        val_check_interval=hparams.val_check_interval,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        profiler=hparams.profiler,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    print(hparams)
    print()
    trainer.fit(model)

    trainer.test()
