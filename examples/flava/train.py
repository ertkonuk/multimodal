# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
import datasets
from data import (
    HFDatasetInfo,
    ImageDataModule,
    MLMDataModule,
    MultiDataModule,
    VLDataModule,
)
import logging
from callbacks.multimodal_eval import MultimodalEvalCallback
from model import FLAVALightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import argumentparser
from  pytorch_lightning.strategies import DDPFullyShardedStrategy

args = argumentparser.get_training_arg_parser().parse_args()
print('Training Arguments:')
print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

# Dataset Config
# ImageNet must be on the disk.
IMAGENET_TRAIN_ROOT = args.imagenet_train_root
IMAGENET_VAL_ROOT   = args.imagenet_val_root
# Set the cache dir for huggingface dataset
datasets.config.DOWNLOADED_DATASETS_PATH = args.hf_dir
datasets.config.HF_DATASETS_CACHE = args.hf_dir
torch.hub.set_dir(args.pyt_dir)

AVAIL_GPUS = args.gpus
SEED = args.seed
NUM_WORKERS = args.num_workers
MAX_STEPS = args.max_steps
BATCH_SIZE = args.batch_size
ALLOW_UNEVEN_BATCHES = args.allow_uneven_batches
CHECKPT_DIR = args.save_dir
NUM_SANITY_VAL_STEPS = args.sanity_steps
PARALLEL_STRATEGY = args.parallel_strategy
vl_kwargs={'num_proc': args.num_proc}
PRECISION = 16 # Not Used


logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

print('-'* 100)
print('dataset downloaddir: ', datasets.config.DOWNLOADED_DATASETS_PATH)
print('dataset cachedir:', datasets.config.HF_DATASETS_CACHE)
print('torch hubdir: ',torch.hub.get_dir())
print('-'* 100)

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
    
    manstrat = DDPFullyShardedStrategy(min_num_params=1e9)

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
	strategy=PARALLEL_STRATEGY
	#plugins="fsdp"
    )
    with torch.autograd.profiler.emit_nvtx():       
        trainer.fit(model, datamodule=datamodule)
    #trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
