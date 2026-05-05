import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
from .models import DigitDataModule, MLPClassifier, CNNClassifier


def run(
    config,
    splits,
    epochs=10,
    progress=False,
    log=False,
    name="mlp",
    model_type="mlp",
    num_workers=15,
):
    dm = DigitDataModule(
        **splits,
        batch_size=config.get("batch_size", 256),
        num_workers=num_workers,
    )

    if model_type == "mlp":
        model = MLPClassifier(**{k: v for k, v in config.items() if k != "batch_size"})
    elif model_type == "cnn":
        model = CNNClassifier(**{k: v for k, v in config.items() if k != "batch_size"})

    callbacks = [
        EarlyStopping(monitor="val_acc", patience=5, verbose=False, mode="max"),
    ]

    if log:
        callbacks += [
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                dirpath=f"checkpoints/{name}",
                filename=f"best-{name}",
                save_top_k=1,
                enable_version_counter=False,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=progress,
        logger=CSVLogger("logs", name=name) if log else False,
        enable_model_summary=False,
    )

    trainer.fit(model, dm)

    if log:
        checkpoint_cb = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
        model = type(model).load_from_checkpoint(checkpoint_cb.best_model_path)

    return trainer, model, dm


def objective(trial, splits, model_type="mlp"):
    if model_type == "mlp":
        hidden_layers = eval(
            trial.suggest_categorical(
                "hidden_layers",
                [
                    "[256, 128]",
                    "[512, 256, 128]",
                    "[1024, 512, 256]",
                    "[512, 256, 64]",
                ],
            )
        )
        activation = trial.suggest_categorical(
            "activation", ["relu", "gelu", "silu", "tanh"]
        )

        norm_layer = trial.suggest_categorical("norm_layer", [None, "BatchNorm1d"])

        optimizer = trial.suggest_categorical(
            "optimizer", ["adam", "adamw", "sgd", "rmsprop"]
        )

        if optimizer == "sgd":
            lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        else:
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        config = {
            "hidden_layers": hidden_layers,
            "activation": activation,
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, log=False),
            "loss_fn": trial.suggest_categorical(
                "loss_fn", ["cross_entropy", "label_smoothing"]
            ),
            "norm_layer": norm_layer,
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256, 512]
            ),
        }
        trainer, _, _ = run(config, splits, epochs=10, model_type="mlp")

    elif model_type == "cnn":
        out_channels = eval(
            trial.suggest_categorical(
                "out_channels",
                [
                    "[32, 64, 128]",
                    "[32, 64, 256]",
                    "[64, 128, 256]",
                ],
            )
        )

        pool_kernel = trial.suggest_categorical("pool_kernel", [2, 3])
        pool_type = trial.suggest_categorical("pool_type", ["max", "avg"])

        norm_layer = {"None": None, "BatchNorm2d": nn.BatchNorm2d}[
            trial.suggest_categorical("norm_layer", ["None", "BatchNorm2d"])
        ]

        optimizer = trial.suggest_categorical(
            "optimizer", ["adam", "adamw", "sgd", "rmsprop"]
        )

        if optimizer == "sgd":
            lr = trial.suggest_float("lr", 1e-2, 1e-1, log=True)
        else:
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        config = {
            "out_channels": out_channels,
            "pool_kernel": pool_kernel,
            "pool_type": pool_type,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, log=False),
            "optimizer": optimizer,
            "loss_fn": trial.suggest_categorical(
                "loss_fn", ["cross_entropy", "label_smoothing"]
            ),
            "norm_layer": norm_layer,
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        }
        trainer, _, _ = run(config, splits, epochs=10, model_type="cnn")

    return float(trainer.callback_metrics.get("val_acc", 0))


def plot_csv_logger(csv_path, model_name):
    metrics = pd.read_csv(csv_path)
    aggreg_metrics = []
    for i, dfg in metrics.groupby("epoch"):
        agg = dict(dfg.mean())
        agg["epoch"] = i
        aggreg_metrics.append(agg)

    df = pd.DataFrame(aggreg_metrics)
    os.makedirs("images", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col in ["train_loss", "val_loss"]:
        if col in df:
            axes[0].plot(df["epoch"], df[col], label=col)
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    for col in ["train_acc", "val_acc"]:
        if col in df:
            axes[1].plot(df["epoch"], df[col], label=col)
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(model_name)
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_curves.png")
    plt.show()
