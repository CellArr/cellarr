import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import tiledb
import cellarr

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_consolidate_symbols():
    np.random.seed(1)

    n = 100
    y = np.eye(n, dtype=int)
    gene_index = [f"gene_{(i % 10)+1}" for i in range(n)]
    cmat, groups = cellarr.utils.utils_anndata.consolidate_duplicate_symbols(
        y, gene_index, consolidate_duplicate_gene_func=sum
    )

    assert len(groups) == 10
    assert cmat.shape[1] == 10
    assert cmat[:, 1].sum() == 10


def test_remap_anndata():
    np.random.seed(1)

    n = 100
    y = np.eye(n, dtype=int)
    gene_index = [f"gene_{(i % 10)+1}" for i in range(n)]

    var_df = pd.DataFrame({"names": gene_index}, index=gene_index)
    obs_df = pd.DataFrame({"cells": [f"cell1_{j+1}" for j in range(n)]})
    adata = anndata.AnnData(layers={"counts": y}, var=var_df, obs=obs_df)

    cmat = cellarr.utils.utils_anndata.remap_anndata(
        adata, feature_set_order={"gene_1": 0, "gene_2": 1}, var_feature_column="index"
    )

    assert isinstance(cmat, dict)
    mat = cmat["counts"]
    assert mat.shape == (100, 2)
    assert len(mat.data) != 0

    cmat = cellarr.utils.utils_anndata.remap_anndata(
        adata, feature_set_order={"gene_1": 0, "gene_2": 1}, var_feature_column="names"
    )

    assert isinstance(cmat, dict)
    mat = cmat["counts"]
    assert mat.shape == (100, 2)
    assert len(mat.data) != 0

    # test with no matching gene symbols should give me a
    # 0 size data array
    cmat = cellarr.utils.utils_anndata.remap_anndata(
        adata, {"gene_10000": 0, "gene_20000": 1}
    )

    assert isinstance(cmat, dict)
    mat = cmat["counts"]
    assert mat.shape == (100, 2)
    assert len(mat.data) == 0

    # test with empty array
    cmat = cellarr.utils.utils_anndata.remap_anndata(adata, {})

    assert isinstance(cmat, dict)
    mat = cmat["counts"]
    assert mat.shape == (100, 0)
    assert len(mat.data) == 0


def test_extract_info():
    np.random.seed(1)

    n = 100
    y = np.eye(n, dtype=int)
    gene_index = [f"gene_{(i % 10)+1}" for i in range(n)]

    var_df = pd.DataFrame({"names": gene_index}, index=gene_index)
    obs_df = pd.DataFrame({"cells": [f"cell1_{j+1}" for j in range(n)]})
    adata = anndata.AnnData(layers={"counts": y}, var=var_df, obs=obs_df)

    cache = cellarr.utils.utils_anndata.extract_anndata_info(
        [adata], obs_subset_columns=["cells", "notexists"]
    )
    assert len(cache) == 1

    gene_symbols = cellarr.utils.utils_anndata.scan_for_features(cache, unique=False)

    assert gene_symbols is not None
    assert len(gene_symbols) == 1
    assert len(gene_symbols[0]) == 100

    ugene_symbols = cellarr.utils.utils_anndata.scan_for_features(cache, unique=True)

    assert ugene_symbols is not None
    assert len(ugene_symbols) == 10

    cell_counts = cellarr.utils.utils_anndata.scan_for_cellcounts(cache)
    assert cell_counts is not None
    assert len(cell_counts) == 1
    assert cell_counts[0] == 100

    cell_meta = cellarr.utils.utils_anndata.scan_for_cellmetadata(cache)
    assert cell_meta is not None
    assert len(cell_meta) == 100
    assert len(cell_meta.columns) == 2
    assert len(cell_meta["notexists"].unique()) == 1
