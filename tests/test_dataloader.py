import tempfile

import anndata
import numpy as np
import os
import pandas as pd
import pytest
import pytorch_lightning as pl
import random
from scipy.sparse import rand
import tiledb
from cellarr import (
    build_cellarrdataset,
    MatrixOptions,
)
from cellarr.ml.dataloader import DataModule
from cellarr.ml.autoencoder import AutoEncoder

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def generate_adata(n, d, study, sample):
    np.random.seed(1)

    y = (100 * rand(n, d, density=0.2, format="csr")).astype(int)

    gene_index = [f"gene_{i+1}" for i in range(d)]
    var_df = pd.DataFrame({"names": gene_index}, index=gene_index)

    obs_df = pd.DataFrame({"cells": [f"cell_{j+1}" for j in range(n)]})
    obs_df["study"] = study
    obs_df["cellarr_sample"] = sample
    labels = [f"label_{i}" for i in range(10)] + [np.nan]
    obs_df["label"] = random.choices(labels, k=n)
    obs_df["index_in_file"] = range(n)

    adata = anndata.AnnData(layers={"counts": y}, var=var_df, obs=obs_df)

    return adata


def test_dataloader():
    tempdir = tempfile.mkdtemp()

    adata1 = generate_adata(1000, 100, study="test1", sample="a")
    adata2 = generate_adata(100, 1000, study="test2", sample="b")
    adata3 = generate_adata(100, 100, study="test3", sample="c")
    adata4 = generate_adata(1000, 100, study="test4", sample="d")
    obs = pd.concat([adata1.obs, adata2.obs, adata3.obs, adata4.obs])
    obs = obs.reset_index(drop=True)

    sample_metadata = obs[["study", "cellarr_sample"]].drop_duplicates()

    build_cellarrdataset(
        output_path=tempdir,
        files=[adata1, adata2, adata3, adata4],
        sample_metadata=sample_metadata,
        cell_metadata=obs,
        matrix_options=MatrixOptions(dtype=np.float32),
    )

    datamodule = DataModule(
        dataset_path=tempdir,
        cell_metadata_uri="cell_metadata",
        gene_annotation_uri="gene_annotation",
        matrix_uri="assays/counts",
        val_studies=["test3"],
        label_column_name="label",
        study_column_name="study",
        sample_column_name="cellarr_sample",
        batch_size=3,
        sample_size=10,
        lognorm=True,
        target_sum=1e4,
        sampling_by_class=True,
        remove_singleton_classes=True,
    )

    assert "test1" in datamodule.train_df["study"].values
    assert "test2" in datamodule.train_df["study"].values
    assert "test3" not in datamodule.train_df["study"].values
    assert "test3" in datamodule.val_df["study"].values

    dataloader = datamodule.train_dataloader()
    batch = next(iter(dataloader))
    data, labels, studies, samples = batch
    assert data.shape == (30, len(datamodule.gene_indices))
    assert len(labels) == 30
    assert len(set(samples)) == 3

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="takes too long locally")
def test_autoencoder():
    tempdir = tempfile.mkdtemp()

    adata1 = generate_adata(1000, 100, study="test1", sample="a")
    adata2 = generate_adata(100, 1000, study="test2", sample="b")
    adata3 = generate_adata(100, 100, study="test3", sample="c")
    adata4 = generate_adata(1000, 100, study="test4", sample="d")
    obs = pd.concat([adata1.obs, adata2.obs, adata3.obs, adata4.obs])
    obs = obs.reset_index(drop=True)

    sample_metadata = obs[["study", "cellarr_sample"]].drop_duplicates()

    build_cellarrdataset(
        output_path=tempdir,
        files=[adata1, adata2, adata3, adata4],
        sample_metadata=sample_metadata,
        cell_metadata=obs,
        matrix_options=MatrixOptions(dtype=np.float32),
    )

    datamodule = DataModule(
        dataset_path=tempdir,
        cell_metadata_uri="cell_metadata",
        gene_annotation_uri="gene_annotation",
        matrix_uri="assays/counts",
        val_studies=["test3"],
        label_column_name="label",
        study_column_name="study",
        sample_column_name="cellarr_sample",
        batch_size=3,
        sample_size=10,
        lognorm=True,
        target_sum=1e4,
        # sampling_by_class=True,
        # remove_singleton_classes=True,
    )

    autoencoder = AutoEncoder(
        n_genes=len(datamodule.gene_indices),
        latent_dim=128,
        hidden_dim=[1024, 1024],
        dropout=0.5,
        input_dropout=0.4,
        residual=False,
    )

    params = {
        "max_epochs": 2,
        "logger": False,
        "log_every_n_steps": 1,
        "limit_train_batches": 4,
    }
    trainer = pl.Trainer(**params)
    trainer.fit(autoencoder, datamodule=datamodule)
    autoencoder.save_all(model_path=os.path.join(tempdir, "test"))
    assert os.path.isfile(os.path.join(tempdir, "test", "encoder.ckpt"))
    assert os.path.isfile(os.path.join(tempdir, "test", "decoder.ckpt"))
