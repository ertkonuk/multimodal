# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict

import torch
from pytorch_lightning import LightningModule
from torchmultimodal.models.flava import (
    flava_model_for_classification,
    flava_model_for_pretraining,
)
from transformers.optimization import get_cosine_schedule_with_warmup

import os

def get_optimizers_for_lightning(
    model: torch.nn.Module,
    learning_rate: float,
    adam_eps: float,
    adam_weight_decay: float,
    warmup_steps: int,
    max_steps: int,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class FLAVALightningModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 450000,
    ):
        super().__init__()
        self.model = flava_model_for_pretraining(pretrained_model_key="flava_full")
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)

        #losses = asdict(output.losses)        
        # added by tugrulkonuk: TODO: put this into a function
        losses = {
                    'mmm_text' : output.losses.mmm_text_loss,
                    'mmm_image' : output.losses.mmm_image_loss,
                    'mim' : output.losses.mim_loss,
                    'mlm' : output.losses.mlm_loss,
                    'itm' : output.losses.itm_loss,
                    'global_CL' : output.losses.global_contrastive_loss
                 }
        
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(f"train/{key}", losses[key], prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        losses = asdict(output.losses)
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(
                    f"validation/losses/{key}", losses[key], prog_bar=True, logger=True
                )

        return total_loss

    def _step(self, batch, batch_idx):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        # print('Batch info:')
        # for key in batch:
        #     print(f'  {key}: {batch[key].shape}')
        output = self.model(
            image=batch.get("image", None),  # bs x 3 x 224 x 224
            image_for_codebook=batch.get("image_for_codebook", None),  # bs x 3 x 112 x 112
            image_patches_mask=batch.get("image_patches_mask", None),  # bs x 14 x 14
            text=batch.get("text", None),  # bs x 77
            text_masked=batch.get("text_masked", None),  # bs x 77
            mlm_labels=batch.get("mlm_labels", None),  # bs x 77
            itm_labels=batch.get("itm_labels", None),  # bs
            required_embedding=required_embedding,
        )
        
        return output

    def configure_optimizers(self):
        return get_optimizers_for_lightning(
            self.model,
            self.learning_rate,
            self.adam_eps,
            self.adam_weight_decay,
            self.warmup_steps,
            self.max_steps,
        )


class FLAVAClassificationLightningModule(FLAVALightningModule):
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 450000,
    ):
        super().__init__()
        self.model = flava_model_for_classification(
            num_classes=num_classes, pretrained_model_key="flava_full"
        )
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        self.log("train/losses/classification", output.loss, prog_bar=True, logger=True)

        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        self.log(
            "validation/losses/classification", output.loss, prog_bar=True, logger=True
        )

        return output.loss

    def _step(self, batch, batch_idx):
        if "image" in batch and ("text" in batch or "text_masked" in batch):
            required_embedding = "mm"
        elif "image" in batch:
            required_embedding = "image"
        elif "text" in batch or "text_masked" in batch:
            required_embedding = "text"
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")

        output = self.model(
            image=batch.get("image", None),
            text=batch.get("text", None),
            required_embedding=required_embedding,
            labels=batch.get("labels", None),
        )

        # TODO: Add accuracy metric to this later.
        return output
