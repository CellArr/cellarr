import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import List
import pytorch_lightning as pl


class Encoder(nn.Module):
    """A class that encapsulates the encoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        residual: bool = False,
    ):
        """Constructor.

        Args:
            n_genes:
                The number of genes in the gene space, representing the input dimensions.

            latent_dim:
                The latent space dimensions

            hidden_dim:
                A list of hidden layer dimensions, describing the number of layers and their dimensions.
                Hidden layers are constructed in the order of the list for the encoder and in reverse
                for the decoder.

            dropout:
                The dropout rate for hidden layers

            input_dropout:
                The dropout rate for the input layer

            residual:
                Use residual connections.
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(n_genes, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x) -> torch.Tensor:
        """Forward.

        Args:
            x: torch.Tensor
                Input tensor corresponding to input layer.

        Returns:
            Output tensor corresponding to output layer.
        """

        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)

    def save_state(self, filename: str):
        """Save model state.

        Args:
            filename:
                Filename to save the model state.
        """

        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Args:
            filename:
                Filename containing the model state.

            use_gpu:
                Boolean indicating whether or not to use GPUs.
        """

        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        self.load_state_dict(ckpt["state_dict"])


class Decoder(nn.Module):
    """A class that encapsulates the decoder."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        residual: bool = False,
    ):
        """Constructor.

        Args:
            n_genes:
                The number of genes in the gene space, representing the input dimensions.

            latent_dim:
                The latent space dimensions

            hidden_dim:
                A list of hidden layer dimensions, describing the number of layers and their dimensions.
                Hidden layers are constructed in the order of the list for the encoder and in reverse
                for the decoder.

            dropout:
                The dropout rate for hidden layers

            residual:
                Use residual connections.
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        if self.residual:
            assert len(set(hidden_dim)) == 1
        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.BatchNorm1d(hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x) -> torch.Tensor:
        """Forward.

        Args:
            x:
                Input tensor corresponding to input layer.

        Returns:
            Output tensor corresponding to output layer.
        """
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1):
                x = layer(x) + x
            else:
                x = layer(x)
        return x

    def save_state(self, filename: str):
        """Save model state.

        Args:
            filename:
                Filename to save the model state.
        """

        torch.save({"state_dict": self.state_dict()}, filename)

    def load_state(self, filename: str, use_gpu: bool = False):
        """Load model state.

        Args:
            filename:
                Filename containing the model state.

            use_gpu:
                Boolean indicating whether or not to use GPUs.
        """

        if not use_gpu:
            ckpt = torch.load(filename, map_location=torch.device("cpu"))
        else:
            ckpt = torch.load(filename)
        self.load_state_dict(ckpt["state_dict"])


class AutoEncoder(pl.LightningModule):
    """A class encapsulating training."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: List[int] = [1024, 1024],
        dropout: float = 0.5,
        input_dropout: float = 0.4,
        lr: float = 5e-3,
        residual: bool = False,
    ):
        """Constructor.

        Args:
            n_genes:
                The number of genes in the gene space, representing the input dimensions.

            latent_dim:
                The latent space dimensions. Defaults to 128.

            hidden_dim:
                A list of hidden layer dimensions, describing the number of layers and their dimensions.
                Hidden layers are constructed in the order of the list for the encoder and in reverse
                for the decoder.

            dropout:
                The dropout rate for hidden layers

            input_dropout:
                The dropout rate for the input layer

            lr:
                The initial learning rate

            residual:
                Use residual connections.
        """

        super().__init__()

        # network architecture
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.lr = lr
        self.residual = residual

        # networks
        self.encoder = Encoder(
            self.n_genes,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            input_dropout=self.input_dropout,
            residual=self.residual,
        )
        self.decoder = Decoder(
            self.n_genes,
            latent_dim=self.latent_dim,
            hidden_dim=list(reversed(self.hidden_dim)),
            dropout=self.dropout,
            residual=self.residual,
        )

        self.mse_loss_fn = nn.MSELoss()
        self.scheduler = None
        self.val_step_outputs = []

    def configure_optimizers(self):
        """Configure optimizers."""

        optimizer = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=0.01)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.001),
            "interval": "epoch",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.scheduler,
        }  # pytorch-lightning required format

    def forward(self, x):
        """Forward.

        Args:
            x:
                Input tensor corresponding to input layer.

        Returns:
            Output tensor corresponding to the last encoder layer.

            Output tensor corresponding to the last decoder layer.
        """

        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def get_loss(self, batch):
        """Calculate the loss.

        Args:
            batch:
                A batch as defined by a pytorch DataLoader.

        Returns:
            The training loss
        """

        cells, labels, studies, samples = batch
        embedding, reconstruction = self(cells)
        return self.mse_loss_fn(cells, reconstruction)

    def training_step(self, batch, batch_idx):
        """Pytorch-lightning training step."""

        loss = self.get_loss(batch)
        self.log("train loss", loss, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        """Pytorch-lightning validation epoch start."""

        super().on_validation_epoch_start()
        self.val_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """Pytorch-lightning validation step."""

        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_step(batch, prefix="val")

    def on_validation_epoch_end(self):
        """Pytorch-lightning validation epoch end evaluation."""

        if self.trainer.datamodule.val_dataset is None:
            return {}
        return self._eval_epoch(prefix="val")

    def _eval_step(self, batch, prefix: str):
        """Evaluation of validation or test step.

        Args:
            batch:
                A batch as defined by a pytorch DataLoader.
            prefix:
                A string prefix to label logs.

        Returns:
            A dictionary containing step evaluation metrics.
        """

        loss = self.get_loss(batch)

        losses = {
            f"{prefix}_loss": loss,
        }

        if prefix == "val":
            self.val_step_outputs.append(losses)
        return losses

    def _eval_epoch(self, prefix: str):
        """Evaluation of validation or test epoch.

        Args:
            prefix:
                A string prefix to label logs.

        Returns:
            A dictionary containing epoch evaluation metrics.
        """

        if prefix == "val":
            step_outputs = self.val_step_outputs

        loss = torch.Tensor([step[f"{prefix}_loss"] for step in step_outputs]).mean()
        self.log(f"{prefix} loss", loss, logger=True)

        losses = {
            f"{prefix}_loss": loss,
        }
        return losses

    def save_all(
        self,
        model_path: str,
    ):
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        # save model
        self.encoder.save_state(os.path.join(model_path, "encoder.ckpt"))
        self.decoder.save_state(os.path.join(model_path, "decoder.ckpt"))

    def load_state(
        self,
        encoder_filename: str,
        decoder_filename: str,
        use_gpu: bool = False,
    ):
        """Load model state.

        Args:
            encoder_filename:
                Filename containing the encoder model state.

            decoder_filename:
                Filename containing the decoder model state.

            use_gpu:
                Boolean indicating whether or not to use GPUs.
        """

        self.encoder.load_state(encoder_filename, use_gpu)
        self.decoder.load_state(decoder_filename, use_gpu)
