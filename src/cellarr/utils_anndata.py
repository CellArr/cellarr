import itertools
from multiprocessing import Pool
from typing import List, Union

import anndata
import numpy as np
from scipy.sparse import coo_matrix, csr_array, csr_matrix

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def remap_anndata(
    h5ad_or_adata: Union[str, anndata.AnnData],
    feature_set_order: dict,
    var_feature_column: str = "index",
    layer_matrix_name: str = "counts",
) -> csr_matrix:
    """Extract and remap the count matrix to the provided feature (gene) set 
    order from the :py:class:`~anndata.AnnData` object.

    Args:
        adata:
            Input ``AnnData`` object.

            Alternatively, may also provide a path to the H5ad file.

            The index of the `var` slot must contain the feature ids
            for the columns in the matrix.

        feature_set_order:
            A dictionary with the feature ids as keys and their index as 
            value (e.g. gene symbols). The feature ids from the 
            ``AnnData`` object are remapped to the feature order from 
            this dictionary.

        var_feature_column:
            Column in ``var`` containing the feature ids (e.g. gene symbols).
            Defaults to the index of the ``var`` slot.

        layer_matrix_name:
            Layer containing the matrix to add to tiledb.
            Defaults to "counts".

    Returns:
        A ``csr_matrix`` representation of the assay matrix.
    """

    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata)
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    mat = adata.layers[layer_matrix_name]

    if var_feature_column == "index":
        symbols = adata.var.index.tolist()
    else:
        symbols = adata.var[var_feature_column].tolist()

    indices_to_keep = [i for i, x in enumerate(symbols) if x in feature_set_order]
    symbols_to_keep = [symbols[i] for i in indices_to_keep]

    mat = mat[:, indices_to_keep].copy()

    indices_to_map = []
    for x in symbols_to_keep:
        indices_to_map.append(feature_set_order[x])

    if isinstance(mat, np.ndarray):
        mat_coo = coo_matrix(mat)
    elif isinstance(mat, (csr_array, csr_matrix)):
        mat_coo = mat.tocoo()
    else:
        raise TypeError(f"Unknown matrix type: {type(mat)}.")

    new_col = np.array([indices_to_map[i] for i in mat_coo.col])

    return coo_matrix(
        (mat_coo.data, (mat_coo.row, new_col)),
        shape=(adata.shape[0], len(feature_set_order)),
    ).tocsr()


def _get_feature_index(
    h5ad_or_adata: Union[str, anndata.AnnData],
    var_feature_column: str = "index",
) -> List[str]:
    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata, backed=True)
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    if var_feature_column == "index":
        symbols = adata.var.index.tolist()
    else:
        symbols = adata.var[var_feature_column].tolist()

    return symbols


def _wrapper_get_feature_ids(args):
    file, gcol = args
    return _get_feature_index(file, gcol)


def scan_for_features(
    h5ad_or_adata: List[Union[str, anndata.AnnData]],
    var_feature_column: str = "index",
    num_threads: int = 1,
) -> List[str]:
    """Extract and generate the list of unique feature identifiers across files.

    Args:
        h5ad_or_adata:
            List of anndata objects or path to h5ad files.

        var_feature_column:
            Column containing the feature ids (e.g. gene symbols).
            Defaults to "index".

        num_threads:
            Number of threads to use.
            Defaults to 1.

    Returns:
        List of all unique feature ids across all files.
    """
    with Pool(num_threads) as p:
        _args = [(file_info, var_feature_column) for file_info in h5ad_or_adata]
        all_symbols = p.map(_wrapper_get_feature_ids, _args)
        return list(set(itertools.chain.from_iterable(all_symbols)))


def _get_cellcounts(h5ad_or_adata: Union[str, anndata.AnnData]) -> int:
    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata, backed=True)
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    return adata.shape[0]


def scan_for_cellcounts(
    h5ad_or_adata: List[Union[str, anndata.AnnData]],
    num_threads: int = 1,
) -> List[int]:
    """Extract cell counts across files.

    Args:
        h5ad_or_adata:
            List of anndata objects or path to h5ad files.

        num_threads:
            Number of threads to use.
            Defaults to 1.

    Returns:
        List of cell counts across files.
    """
    with Pool(num_threads) as p:
        all_counts = p.map(_get_cellcounts, h5ad_or_adata)
        return all_counts
