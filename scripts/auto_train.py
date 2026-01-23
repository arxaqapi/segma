import argparse
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Literal

import lightning as pl
import torch
import torch._dynamo.config
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from segma.config import Config, load_config
from segma.data import SegmaFileDataset, SegmentationDataLoader
from segma.models import (
    HydraWhisper,
    Models,
    SurgicalHydraHubert,
    SurgicalWhisper,
)
from segma.utils import set_seed
from segma.utils.encoders import MultiLabelEncoder
from segma.utils.experiment import new_experiment_id


def get_parameter_table(model: torch.nn.Module):
    m_length = max(len(name) for name, _ in model.named_parameters()) + 2
    total_params = 0
    total_trainable_params = 0

    print(f"{'Name':<{m_length}} | {'Params':<10} | {'Trainable params':<10}")
    print("-" * (m_length + 32))
    for name, p in model.named_parameters():
        np = p.numel()
        ntp = p.numel() if p.requires_grad else 0

        total_params += np
        total_trainable_params += ntp

        print(f"{name:<{m_length}} | {np:<10} | {ntp:<10}")

    # TODO - add percent trainable
    print(total_params, total_trainable_params)


def get_metric(metric: str) -> tuple[Literal["min", "max"], str]:
    match metric:
        case "loss":
            return "min", "val/loss"
        case "f1_score":
            return "max", "val/f1_score"
        case "auroc":
            return "max", "val/auroc"
        case _:
            raise ValueError(
                f"metric '{metric}' is not supported, please use 'loss', 'auroc' or 'f1_score'."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file to be loaded and used for the training.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Tags to be added to the wandb logging instance.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume training, pass in checkpoint path.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Finetune all weights of the model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments",
        help="Output path of the model artifacts.",
    )
    parser.add_argument("--run-id", "--id", type=str, help="ID of the run")

    args, extra_args = parser.parse_known_args()

    if args.auto_resume and not args.run_id:
        raise ValueError("When passing auto-resume, please add a valid run-id")
    if not args.run_id:
        args.run_id = new_experiment_id()

    config: Config = load_config(config_path=args.config, cli_extra_args=extra_args)
    if config.train.seed:
        set_seed(config.train.seed)

    experiment_path = Path(args.output)
    if not experiment_path.exists():
        experiment_path.mkdir()

    save_path = experiment_path / args.run_id
    save_path.mkdir(parents=True, exist_ok=True)
    config.save(save_path / "config.yml")

    chkp_path = save_path / "checkpoints"
    chkp_path.mkdir(parents=True, exist_ok=True)
    last_ckpt = chkp_path / "last.ckpt"

    if "hydra" not in config.model.name:
        raise ValueError("Only `MultiLabelEncoder` is supported")
    l_encoder = MultiLabelEncoder(labels=config.data.classes)

    model: SurgicalWhisper | HydraWhisper | SurgicalHydraHubert = Models[
        config.model.name
    ](l_encoder, config)

    mode, monitor = get_metric(config.train.validation_metric)

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=config.train.lr)
        return {
            "optimizer": optim,
            "lr_scheduler": ReduceLROnPlateau(
                optim, mode=mode, patience=config.train.scheduler.patience
            ),
            "monitor": monitor,
        }

    model.configure_optimizers = MethodType(configure_optimizers, model)

    # Somewhere not syncing good params
    if config.model.name == "surgical_hubert_hydra":
        if not config.model.config.freeze_encoder:
            for p in model.wav2vec2.encoder.parameters():
                p.requires_grad = True

    get_parameter_table(model)

    print("segmafile loading")
    sfd = SegmaFileDataset.from_config(config)

    sfd.load(use_cache=False)

    print(
        f"[log @ {datetime.now().strftime('%Y%m%d_%H:%M:%S')}] - SegmentationDataLoader initializing ...",
        flush=True,
    )
    dm = SegmentationDataLoader(
        dataset=sfd,
        label_encoder=l_encoder,
        config=config,
        conv_settings=model.conv_settings,
        audio_preparation_hook=model.audio_preparation_hook,
    )
    print(
        f"[log @ {datetime.now().strftime('%Y%m%d_%H:%M:%S')}] - SegmentationDataLoader initialized",
        flush=True,
    )

    print("[log] - use WandbLogger")
    logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name,
        id=args.run_id.split("-")[-1],
        log_model=False if config.wandb.offline else "all",
        tags=args.tags,
        offline=config.wandb.offline,
        resume="must" if args.auto_resume and last_ckpt.exists() else None,  # "never",
    )
    # Allow val_change maybe not best idea but needed it for some reason
    # TODO
    logger.experiment.config.update(config, allow_val_change=True)
    save_path = save_path.with_stem(save_path.stem + f"-{logger.experiment.id}")

    model_checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=10,
        # every_n_epochs=1,
        save_last=True,
        dirpath=chkp_path,
        filename="epoch={epoch:02d}-val_loss={val/loss:.3f}",
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=False,
    )

    # NOTE - from scratch training and resume
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epochs,
        logger=logger,
        callbacks=[
            model_checkpoint,
            early_stopping,
            LearningRateMonitor(),
            TQDMProgressBar(1000 if "debug" not in config.data.dataset_path else 1),
        ],
        # profiler="advanced"
        profiler=config.train.profiler,
    )

    # https://pytorch.org/docs/main/torch.compiler_troubleshooting.html#dealing-with-recompilations
    # https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.t130sdb4rshr
    torch._dynamo.config.accumulated_cache_size_limit = 32
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 32
    model = torch.compile(model)

    print(f"[log @ {datetime.now().strftime('%Y%m%d_%H%M')}] - started training")
    if args.auto_resume and last_ckpt.exists():
        print("[log] - fit with resuming")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.fit(model, datamodule=dm)

    # NOTE - symlink to best model and to static best model (models/last/best.ckpt)
    (chkp_path / "best.ckpt").symlink_to(
        Path(model_checkpoint.best_model_path).absolute()
    )
    static_p = experiment_path / "last"
    static_p.mkdir(parents=True, exist_ok=True)
    bm_static_p = static_p / "best.ckpt"
    bm_static_p.unlink(missing_ok=True)
    bm_static_p.symlink_to(Path(model_checkpoint.best_model_path).absolute())

    print(f"[log] - best model score: {model_checkpoint.best_model_score}")
    print(f"[log] - best model path: {model_checkpoint.best_model_path}")
