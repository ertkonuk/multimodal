# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from data import (
    HFDatasetInfo,
    ImageDataModule,
    MLMDataModule,
    MultiDataModule,
    VLDataModule,
)
#from examples.flava.callbacks.multimodal_eval import MultimodalEvalCallback
from callbacks.multimodal_eval import MultimodalEvalCallback
from model import FLAVALightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor


AVAIL_GPUS = -1
SEED = -1

# ImageNet must be on the disk.
IMAGENET_TRAIN_ROOT = "/datasets/imagenet-raw/raw-data/val"
IMAGENET_VAL_ROOT   = "/datasets/imagenet-raw/raw-data/val"

NUM_WORKERS = 16
MAX_STEPS = 250 #450000
BATCH_SIZE = 48
ALLOW_UNEVEN_BATCHES = False
PRECISION = 16

# added by tugrulkonuk
# Change the download folder and the cache_dir for huggingface datasets package to
# avoid downloading to the home folder (default)
import datasets, os
from pathlib import Path

# No need Path but I am too lazy to fix it
# Set the checkpoint directory
CHECKPT_DIR = Path('/tmp/mm')

# Set the cache dir for huggingface transformers
#os.environ['TRANSFORMERS_CACHE'] = Path('/tmp/multimodal')

# Set the cache dir for huggingface dataset
datasets.config.DOWNLOADED_DATASETS_PATH = Path('/datasets/tkonuk')
datasets.config.HF_DATASETS_CACHE = Path('/datasets/tkonuk')

# Set the cache dir for pytorch: SET TO THI A MORE APPROPRIATE DIRECTORY
#torch.hub.set_dir(Path('/datasets/tkonuk'))
torch.hub.set_dir(Path('/tmp/mm'))

print('-'* 100)
print('transformers cachedir: ', os.environ['TRANSFORMERS_CACHE'])
print('dataset downloaddir: ', datasets.config.DOWNLOADED_DATASETS_PATH)
print('dataset cachedir:', datasets.config.HF_DATASETS_CACHE)
print('torch hubdir: ',torch.hub.get_dir())
print('-'* 100)

# Number of threads for data downloading
NUM_PROC = 32
vl_kwargs={'num_proc': NUM_PROC}

NUM_SANITY_VAL_STEPS = 0
PARALLEL_STRATEGY = "ddp" #"ddp_sharded"

import logging
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

def main():
    if SEED != -1:
        seed_everything(SEED, workers=True)
    
    imagenet_datamodule = ImageDataModule(
        train_root=IMAGENET_TRAIN_ROOT,
        val_root=IMAGENET_VAL_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )

    mlm_datamodule = MLMDataModule(
        [HFDatasetInfo("wikitext", "wikitext-103-raw-v1")],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )

    vl_datamodule = MultiDataModule(
        [
            VLDataModule(
                train_dataset_infos=[
                    HFDatasetInfo(
                        key="red_caps",
                        subset="cupcakes",
                        rename_columns=[("caption", "text")],
                    )
                ],
                val_dataset_infos=[
                    HFDatasetInfo(
                        key="red_caps",
                        subset="cupcakes",
                        rename_columns=[("caption", "text")],
                        split_key_mapping={"validation": "train"},
                    )
                ],
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
            )
        ]
    )

    datamodule = MultiDataModule([imagenet_datamodule, mlm_datamodule, vl_datamodule])
    
    datamodule.setup("fit")
    
    model = FLAVALightningModule()
    
    trainer = Trainer(
        default_root_dir=CHECKPT_DIR,
        num_sanity_val_steps=NUM_SANITY_VAL_STEPS,
        max_steps=MAX_STEPS,
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=50,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            MultimodalEvalCallback(imagenet_datamodule=imagenet_datamodule),
        ],
        strategy=PARALLEL_STRATEGY,
    )
           
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
