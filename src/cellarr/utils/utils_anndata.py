import itertools
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, Union

import anndata
import mopsy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_array, csc_matrix, csr_array, csr_matrix

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def remap_anndata(
    h5ad_or_adata: Union[str, anndata.AnnData],
    feature_set_order: dict,
    var_feature_column: str = "index",
    layer_matrix_name: Union[str, List[str]] = "counts",
    consolidate_duplicate_gene_func: Union[callable, List[callable]] = sum,
) -> Dict[str, csr_matrix]:
    """Extract and remap the count matrix to the provided feature (gene) set order from the :py:class:`~anndata.AnnData`
    object.

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
            Layer containing the matrix to add to TileDB.
            Defaults to "counts".

            Alternatively, may provide a list of layers to extract and add
            to TileDB.

        consolidate_duplicate_gene_func:
            Function to consolidate when the `AnnData` object contains
            multiple rows with the same feature id or gene symbol.

            Defaults to :py:func:`sum`.

    Returns:
        A dictionary with the key containing the name of the layer
        and the output a ``csr_matrix`` representation of the assay matrix.
    """

    if isinstance(layer_matrix_name, str):
        layer_matrix_name = [layer_matrix_name]

    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata, "r")
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    if len(feature_set_order) == 0:
        omat = {}
        for lmn in layer_matrix_name:
            omat[lmn] = coo_matrix(
                ([], ([], [])),
                shape=(adata.shape[0], len(feature_set_order)),
            ).tocsr()
        return omat

    if var_feature_column == "index":
        osymbols = adata.var.index.tolist()
    else:
        osymbols = adata.var[var_feature_column].tolist()

    omat = {}
    counter = 0
    for lmn in layer_matrix_name:
        _consol_func = consolidate_duplicate_gene_func
        if isinstance(_consol_func, list):
            _consol_func = consolidate_duplicate_gene_func[counter]

        mat, symbols = consolidate_duplicate_symbols(
            adata.layers[lmn],
            feature_ids=osymbols,
            consolidate_duplicate_gene_func=_consol_func,
        )

        counter += 1
        # figure out which indices to keep from the matrix
        indices_to_keep = [i for i, x in enumerate(symbols) if x in feature_set_order]
        symbols_to_keep = [symbols[i] for i in indices_to_keep]

        mat = mat[:, indices_to_keep].copy()

        if len(indices_to_keep) == 0:
            omat[lmn] = coo_matrix(
                ([], ([], [])),
                shape=(adata.shape[0], len(feature_set_order)),
            ).tocsr()
            continue

        # figure out mapping from the current indices to the original feature order
        indices_to_map = []
        for x in symbols_to_keep:
            indices_to_map.append(feature_set_order[x])

        if isinstance(mat, np.ndarray):
            mat_coo = coo_matrix(mat)
        elif isinstance(mat, (csr_array, csr_matrix, csc_matrix, csc_array)):
            mat_coo = mat.tocoo()
        else:
            raise TypeError(f"Unknown matrix type: {type(mat)}.")

        # remap gene symbols to the new feature order
        new_col = np.array([indices_to_map[i] for i in mat_coo.col])

        omat[lmn] = coo_matrix(
            (mat_coo.data, (mat_coo.row, new_col)),
            shape=(adata.shape[0], len(feature_set_order)),
        ).tocsr()

    return omat


def consolidate_duplicate_symbols(
    matrix: Any, feature_ids: List[str], consolidate_duplicate_gene_func: callable
) -> anndata.AnnData:
    """Consolidate duplicate gene symbols.

    Args:
        matrix:
            data matrix with rows
            for cells and columns for genes.

        feature_ids:
            List of feature ids along the column axis of the matrix.

        consolidate_duplicate_gene_func:
            Function to consolidate when the AnnData object contains
            multiple rows with the same feature id or gene symbol.

            Defaults to :py:func:`sum`.

    Returns:
        AnnData object with duplicate gene symbols consolidated.
    """

    if len(set(feature_ids)) == len(feature_ids):
        return matrix, feature_ids

    return mopsy.apply(consolidate_duplicate_gene_func, mat=matrix, group=feature_ids, axis=1)


def _sanitize_frame_with_missing_cols(frame, expected_columns, num_cells):
    output = frame.copy()
    if expected_columns is not None and len(expected_columns) > 0:
        _common = list(set(expected_columns).intersection(frame.columns))
        output = pd.DataFrame({})

        if len(_common) > 0:
            output = frame[_common]

        for col in expected_columns:
            if col not in output.columns:
                output[col] = ["NA"] * num_cells

    return output


def _extract_info(
    h5ad_or_adata: Union[str, anndata.AnnData],
    var_feature_column: str = "index",
    var_subset_columns: List[str] = None,
    obs_subset_columns: List[str] = None,
) -> Tuple[List[str], pd.DataFrame, int]:
    if isinstance(h5ad_or_adata, str):
        adata = anndata.read_h5ad(h5ad_or_adata, "r")
    else:
        if not isinstance(h5ad_or_adata, anndata.AnnData):
            raise TypeError("Input is not an 'AnnData' object.")

        adata = h5ad_or_adata

    if var_feature_column == "index":
        symbols = adata.var.index.tolist()
    else:
        symbols = adata.var[var_feature_column].tolist()

    symbols_df = adata.var.copy()
    symbols_df.index = symbols

    symbols_df = _sanitize_frame_with_missing_cols(symbols_df, var_subset_columns, adata.shape[1])
    features_df = _sanitize_frame_with_missing_cols(adata.obs, obs_subset_columns, adata.shape[0])

    return symbols_df, features_df, adata.shape[0]


def _wrapper_extract_info(args):
    file, gcol, gcols, ccols = args
    return _extract_info(file, gcol, gcols, ccols)


def extract_anndata_info(
    h5ad_or_adata: List[Union[str, anndata.AnnData]],
    var_feature_column: str = "index",
    var_subset_columns: List[str] = None,
    obs_subset_columns: dict = None,
    num_threads: int = 1,
):
    """Extract and generate the list of unique feature identifiers and cell counts across files.

    Args:
        h5ad_or_adata:
            List of anndata objects or path to h5ad files.

        var_feature_column:
            Column containing the feature ids (e.g. gene symbols).
            Defaults to "index".

        var_subset_columns:
            List of var columns to concatenate across all files.
            Defaults to None and no metadata columns will be extracted.

        obs_subset_columns:
            List of obs columns to concatenate across all files.
            Defaults to None and no metadata columns will be extracted.

        num_threads:
            Number of threads to use.
            Defaults to 1.
    """
    with Pool(num_threads) as p:
        _args = [(file_info, var_feature_column, var_subset_columns, obs_subset_columns) for file_info in h5ad_or_adata]
        return p.map(_wrapper_extract_info, _args)


def scan_for_features(cache, unique: bool = True) -> List[str]:
    """Extract and generate the list of unique feature identifiers across files.

    Needs calling :py:func:`~.extract_anndata_info` first.

    Args:
        cache:
            Info extracted by typically running
            :py:func:`~.extract_anndata_info`.

        unique:
            Compute gene list to a unique list.

    Returns:
        List of all unique feature ids across all files.
    """
    _features = [x[0].index.tolist() for x in cache]
    if unique:
        return list(set(itertools.chain.from_iterable(_features)))

    return _features


def scan_for_features_annotations(cache, unique: bool = True) -> List[str]:
    """Extract and generate feature annotation metadata across all files in cache.

    Needs calling :py:func:`~.extract_anndata_info` first.

    Args:
        cache:
            Info extracted by typically running
            :py:func:`~.extract_anndata_info`.

        unique:
            Compute gene list to a unique list.

    Returns:
        List of all unique feature ids across all files.
    """

    _featmeta = pd.concat([x[1] for x in cache])
    return _featmeta


def scan_for_cellcounts(cache) -> List[int]:
    """Extract cell counts across files.

    Needs calling :py:func:`~.extract_anndata_info` first.

    Args:
        cache:
            Info extracted by typically running
            :py:func:`~.extract_anndata_info`.

    Returns:
        List of cell counts across files.
    """

    _cellcounts = [x[2] for x in cache]
    return _cellcounts


def scan_for_cellmetadata(cache) -> List[int]:
    """Extract and merge all cell metadata data frames across files.

    Needs calling :py:func:`~.extract_anndata_info` first.

    Args:
        cache:
            Info extracted by typically running
            :py:func:`~.extract_anndata_info`.

    Returns:
        A :py:class:`pandas.Dataframe` containing all cell metadata.
    """

    _cellmeta = pd.concat([x[1] for x in cache], ignore_index=True)
    return _cellmeta
