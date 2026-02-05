import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.classification import binary_f1_score

from segma.config import TrainConfig


class MultiLabelModel(L.LightningModule):
    def __init__(self, model: nn.Module, train_config: TrainConfig):
        super().__init__()

        self.model = model
        self.train_config = train_config
        self.label_encoder = self.model.label_encoder

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.train_config.learning_rate)
        return {
            "optimizer": optim,
            "lr_scheduler": ReduceLROnPlateau(
                optim, mode="min", patience=self.train_config.scheduler_patience
            ),
            "monitor": "val/loss",
        }

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        y_pred_heads = self.forward(x, output="dict")

        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.model.label_encoder.labels)  # ty: ignore
        y_target = y_target.view(-1, n_labels)
        # (batch * n_windows) - flattened, usefull when slicing target vector at the end
        y_preds = {k: y_pred.view(-1) for k, y_pred in y_pred_heads.items()}

        head_losses = {
            k: torch.nn.functional.binary_cross_entropy_with_logits(
                input=y_pred, target=y_target[..., i]
            )
            for i, (k, y_pred) in enumerate(y_preds.items())
        }

        loss = torch.stack(list(head_losses.values())).sum()
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        for head_name, head_loss in head_losses.items():
            self.log(
                f"train/loss_{head_name.removeprefix('linear_head_')}",
                head_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y_target = batch["y"]
        y_pred_heads = self.forward(x, output="dict")

        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.model.label_encoder.labels)

        y_target = y_target.view(-1, n_labels)
        # (batch * n_windows) - flattened, usefull when slicing target vector at the end
        y_preds = {k: y_pred.view(-1) for k, y_pred in y_pred_heads.items()}

        # NOTE - loss computation
        head_losses = {
            k: torch.nn.functional.binary_cross_entropy_with_logits(
                input=y_pred, target=y_target[..., i]
            )
            for i, (k, y_pred) in enumerate(y_preds.items())
        }

        loss = torch.stack(list(head_losses.values())).sum()
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for head_name, head_loss in head_losses.items():
            self.log(
                f"val/loss_{head_name.removeprefix('linear_head_')}",
                head_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        # NOTE - f1 score
        head_f1_scores = {
            k: binary_f1_score(
                preds=y_pred,
                target=y_target[..., i],
                threshold=0.5,
            )
            for i, (k, y_pred) in enumerate(y_preds.items())
        }
        for head_name, head_f1_score in head_f1_scores.items():
            self.log(
                f"val/f1_{head_name.removeprefix('linear_head_')}",
                head_f1_score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
