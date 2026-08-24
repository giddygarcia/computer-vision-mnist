import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW, SGD, RMSprop


class DigitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=256,
        num_workers=15,
    ):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _ds(self, X, y):
        return TensorDataset(torch.tensor(X), torch.tensor(y))

    def train_dataloader(self):
        return DataLoader(
            self._ds(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self._ds(self.X_val, self.y_val),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self._ds(self.X_test, self.y_test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0
        )


class MLPClassifier(pl.LightningModule):
    def __init__(
        self,
        hidden_layers=[512, 256, 128],
        activation="relu",
        dropout=0.3,
        optimizer="adam",
        lr=1e-3,
        loss_fn="cross_entropy",
        norm_layer=None,
        weight_decay=0.0,
    ):
        super().__init__()

        norm_layer = {nn.BatchNorm1d: "BatchNorm1d", None: None}.get(
            norm_layer, norm_layer
        )

        self.save_hyperparameters()

        norm_cls = {"BatchNorm1d": nn.BatchNorm1d, "None": None, None: None}.get(
            self.hparams.norm_layer
        )

        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        loss_fns = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "label_smoothing": nn.CrossEntropyLoss(label_smoothing=0.1),
        }
        self.loss_fn = loss_fns[loss_fn]

        layers, in_dim = [], 784
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            if norm_cls is not None:
                layers.append(norm_cls(h))
            layers.append(acts[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def configure_optimizers(self):
        opts = {"adam": Adam, "adamw": AdamW, "sgd": SGD, "rmsprop": RMSprop}
        extra = {"momentum": 0.9} if self.hparams.optimizer == "sgd" else {}
        return opts[self.hparams.optimizer](
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            **extra,
        )

class CNNClassifier(pl.LightningModule):
    def __init__(
        self,
        out_channels=[32, 64],
        lr=1e-3,
        dropout=0.3,
        optimizer="adam",
        loss_fn="cross_entropy",
        norm_layer=nn.BatchNorm2d,
        pool_kernel=2,
        pool_type="max",
        weight_decay=0.0,
    ):
        super().__init__()

        norm_layer = {nn.BatchNorm2d: "BatchNorm2d", None: None}.get(
            norm_layer, norm_layer
        )
        self.save_hyperparameters()

        norm_cls = {
            "BatchNorm2d": nn.BatchNorm2d,
            "None": None,
            None: None,
        }.get(self.hparams.norm_layer)

        loss_fns = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "label_smoothing": nn.CrossEntropyLoss(label_smoothing=0.1),
        }
        self.loss_fn = loss_fns[loss_fn]

        cnn_layers = []
        in_channels = 1

        pool_cls = {
            "max": nn.MaxPool2d,
            "avg": nn.AvgPool2d,
        }[self.hparams.pool_type]

        for out_channel in out_channels:
            cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channel,
                    kernel_size=3,
                    padding=1,
                )
            )

            if norm_cls is not None:
                cnn_layers.append(norm_cls(out_channel))

            cnn_layers.append(nn.ReLU())

            cnn_layers.append(
                pool_cls(
                    self.hparams.pool_kernel,
                    stride=self.hparams.pool_kernel,
                )
            )

            in_channels = out_channel

        self.cnn_layers = nn.Sequential(*cnn_layers)

        dummy = torch.zeros(1, 1, 28, 28)
        flat_size = self.cnn_layers(dummy).view(1, -1).shape[1]

        fc_layers = [nn.Linear(flat_size, 128)]

        if norm_cls is not None:
            fc_layers.append(nn.BatchNorm1d(128))

        fc_layers.append(nn.ReLU())

        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))

        fc_layers.append(nn.Linear(128, 10))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.cnn_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_layers(x)

    def _step(self, batch, stage):
        x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        self.log(
            f"{stage}_acc",
            acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def test_step(self, batch, _):
        return self._step(batch, "test")

    def configure_optimizers(self):
        opts = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
            "rmsprop": RMSprop,
        }

        extra = {"momentum": 0.9} if self.hparams.optimizer == "sgd" else {}

        return opts[self.hparams.optimizer](
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            **extra,
        )