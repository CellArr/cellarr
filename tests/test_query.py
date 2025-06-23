import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import tiledb
from cellarr import CellArrDataset, build_cellarrdataset, MatrixOptions

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def generate_adata(n, d, k):
    np.random.seed(1)

    z = np.random.normal(loc=np.arange(k), scale=np.arange(k) * 2, size=(n, k))
    w = np.random.normal(size=(d, k))
    y = np.dot(z, w.T)

    gene_index = [f"gene_{i+1}" for i in range(d)]
    var_df = pd.DataFrame({"names": gene_index}, index=gene_index)
    obs_df = pd.DataFrame({"cells": [f"cell1_{j+1}" for j in range(n)]})

    adata = anndata.AnnData(layers={"counts": y}, var=var_df, obs=obs_df)

    return adata


def test_query_cellarrdataset():
    tempdir = tempfile.mkdtemp()

    adata1 = generate_adata(1000, 100, 10)
    adata2 = generate_adata(100, 1000, 100)

    dataset = build_cellarrdataset(
        output_path=tempdir,
        files=[adata1, adata2],
        matrix_options=MatrixOptions(dtype=np.float32),
    )

    assert dataset is not None
    assert isinstance(dataset, CellArrDataset)

    cd = CellArrDataset(dataset_path=tempdir)

    gene_list = ["gene_1", "gene_95", "gene_50"]

    gfp = tiledb.open(f"{tempdir}/gene_annotation", "r")

    genes = gfp.df[:]

    adata1_gene_indices = sorted(
        [adata1.var.index.tolist().index(x) for x in gene_list]
    )

    adata2_gene_indices = sorted(
        [adata2.var.index.tolist().index(x) for x in gene_list]
    )

    result1 = cd[0, gene_list]

    assert result1 is not None
    assert result1.matrix["counts"].shape == (1, len(adata1_gene_indices))

    assert np.allclose(
        result1.matrix["counts"].data,
        adata1.layers["counts"][0, adata1_gene_indices],
    )

    result2 = cd[1000, gene_list]
    assert result2.matrix["counts"].shape == (1, len(adata2_gene_indices))

    assert np.allclose(
        result2.matrix["counts"].data,
        adata2.layers["counts"][0, adata2_gene_indices],
    )

    assert cd.shape == (1100, 1000)
    assert len(cd) == 1100

    assert cd.get_cell_metadata_columns() == [
        "cellarr_sample",
        "cellarr_cell_index_in_sample",
    ]
    assert len(cd.get_cell_metadata_column("cellarr_sample")) == 1100
    assert len(cd.get_cell_subset("cellarr_sample == 'sample_1'")) == 1000

    assert sorted(cd.get_sample_metadata_columns()) == sorted(
        ["cellarr_sample", "cellarr_original_gene_set", "cellarr_cell_counts", "cellarr_sample_end_index", "cellarr_sample_start_index"]
    )
    assert len(cd.get_sample_metadata_column("cellarr_sample")) == 2
    assert len(cd.get_sample_subset("cellarr_sample == 'sample_1'")) == 1

    assert cd.get_gene_annotation_columns() == ["cellarr_gene_index"]
    assert len(cd.get_gene_annotation_column("cellarr_gene_index")) == 1000
    assert len(cd.get_gene_subset("cellarr_gene_index == 'gene_1'")) == 1

    result1 = cd.get_slice(slice(0, 100), gene_list)

    assert result1 is not None
    assert result1.matrix["counts"].shape == (101, len(gene_list))

    assert result1.to_anndata() is not None
    assert result1.to_summarizedexperiment() is not None

    assert cd.get_cells_for_sample(0).to_anndata().shape == (adata1.shape[0], 1000)
    assert cd.get_cells_for_sample(1).to_anndata().shape == (adata2.shape[0], 1000)

    sample_count = 0
    obs = [adata1, adata2]
    for sample, sample_chunk in cd.itersamples():
        assert len(sample_chunk.cell_metadata) == obs[sample_count].shape[0]
        sample_count += 1

    assert sample_count == 2

    cell_count = 0
    for cell, cell_chunk in cd.itercells():
        cell_count += 1
    assert cell_count == 1100

    full_row_slice = cd[1,]
    assert full_row_slice is not None
    assert full_row_slice.shape == (1, 1000)

    full_col_slice = cd[:, 1]
    assert full_col_slice is not None
    assert full_col_slice.shape == (1100, 1)


def test_query_cellarrdataset_full_path():
    tempdir = tempfile.mkdtemp()

    adata1 = generate_adata(1000, 100, 10)
    adata2 = generate_adata(100, 1000, 100)

    dataset = build_cellarrdataset(
        output_path=tempdir,
        files=[adata1, adata2],
        matrix_options=MatrixOptions(dtype=np.float32),
    )

    assert dataset is not None
    assert isinstance(dataset, CellArrDataset)

    cd = CellArrDataset.initialize_from_paths(
        gene_annotation_uri=f"{tempdir}/gene_annotation",
        cell_metadata_uri=f"{tempdir}/cell_metadata",
        sample_metadata_uri=f"{tempdir}/sample_metadata",
        assay_uri=f"{tempdir}/assays/counts"
    )

    gene_list = ["gene_1", "gene_95", "gene_50"]

    gfp = tiledb.open(f"{tempdir}/gene_annotation", "r")

    genes = gfp.df[:]

    adata1_gene_indices = sorted(
        [adata1.var.index.tolist().index(x) for x in gene_list]
    )

    adata2_gene_indices = sorted(
        [adata2.var.index.tolist().index(x) for x in gene_list]
    )

    result1 = cd[0, gene_list]

    print(result1)

    assert result1 is not None
    assert result1.matrix[f"{tempdir}/assays/counts"].shape == (1, len(adata1_gene_indices))