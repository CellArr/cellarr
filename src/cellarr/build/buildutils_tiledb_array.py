import os
import shutil
from typing import Union

import numpy as np
import tiledb
from scipy.sparse import csr_array, csr_matrix

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def create_tiledb_array(
    tiledb_uri_path: str,
    x_dim_length: int = None,
    y_dim_length: int = None,
    x_dim_name: str = "cell_index",
    y_dim_name: str = "gene_index",
    matrix_attr_name: str = "data",
    x_dim_dtype: np.dtype = np.uint32,
    y_dim_dtype: np.dtype = np.uint32,
    matrix_dim_dtype: np.dtype = np.uint32,
    is_sparse: bool = True,
):
    """Create a TileDB file with the provided attributes to persistent storage.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the array TileDB file.

        x_dim_length:
            Number of entries along the x/fastest-changing dimension.
            e.g. Number of cells.
            Defaults to None, in which case, the max integer value of
            ``x_dim_dtype`` is used.

        y_dim_length:
            Number of entries along the y dimension.
            e.g. Number of genes.
            Defaults to None, in which case, the max integer value of
            ``y_dim_dtype`` is used.

        x_dim_name:
            Name for the x-dimension.
            Defaults to "cell_index".

        y_dim_name:
            Name for the y-dimension.
            Defaults to "gene_index".

        matrix_attr_name:
            Name for the attribute in the array.
            Defaults to "data".

        x_dim_dtype:
            NumPy dtype for the x-dimension.
            Defaults to np.uint32.

        y_dim_dtype:
            NumPy dtype for the y-dimension.
            Defaults to np.uint32.

        matrix_dim_dtype:
            NumPy dtype for the values in the matrix.
            Defaults to np.uint32.

        is_sparse:
            Whether the matrix is sparse.
            Defaults to True.
    """

    if x_dim_length is None:
        x_dim_length = np.iinfo(x_dim_dtype).max

    if y_dim_length is None:
        y_dim_length = np.iinfo(y_dim_dtype).max

    xdim = tiledb.Dim(name=x_dim_name, domain=(0, x_dim_length - 1), dtype=x_dim_dtype)
    ydim = tiledb.Dim(name=y_dim_name, domain=(0, y_dim_length - 1), dtype=y_dim_dtype)

    dom = tiledb.Domain(xdim, ydim)

    # expecting counts
    tdb_attr = tiledb.Attr(
        name=matrix_attr_name,
        dtype=matrix_dim_dtype,
        filters=tiledb.FilterList([tiledb.GzipFilter()]),
    )

    schema = tiledb.ArraySchema(domain=dom, sparse=is_sparse, attrs=[tdb_attr])

    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    tiledb.Array.create(tiledb_uri_path, schema)

    tdbfile = tiledb.open(tiledb_uri_path, "w")
    tdbfile.close()


def write_csr_matrix_to_tiledb(
    tiledb_array_uri: Union[str, tiledb.SparseArray],
    matrix: csr_matrix,
    value_dtype: np.dtype = np.uint32,
    row_offset: int = 0,
    batch_size: int = 25000,
):
    """Append and save a :py:class:`~scipy.sparse.csr_matrix` to TileDB.

    Args:
        tiledb_array_uri:
            TileDB array object or path to a TileDB object.

        matrix:
            Input matrix to write to TileDB, must be a
            :py:class:`~scipy.sparse.csr_matrix` matrix.

        value_dtype:
            NumPy dtype to reformat the matrix values.
            Defaults to ``uint32``.

        row_offset:
            Offset row number to append to matrix.
            Defaults to 0.

        batch_size:
            Batch size.
            Defaults to 25000.
    """
    tiledb_fp = tiledb_array_uri
    if isinstance(tiledb_array_uri, str):
        tiledb_fp = tiledb.open(tiledb_array_uri, "w")

    if not isinstance(matrix, (csr_array, csr_matrix)):
        raise TypeError("sparse matrix must be in csr format.")

    indptrs = matrix.indptr
    indices = matrix.indices
    data = matrix.data.astype(value_dtype)

    if matrix.shape[1] == 0 or len(data) == 0:
        return

    x = []
    y = []
    vals = []
    for i, indptr in enumerate(indptrs):
        if i != 0 and (i % batch_size == 0 or i == len(indptrs) - 1):
            tiledb_fp[x, y] = vals
            x = []
            y = []
            vals = []

        stop = None
        if i != len(indptrs) - 1:
            stop = indptrs[i + 1]

        val_slice = data[slice(indptr, stop)]
        ind_slice = indices[slice(indptr, stop)]

        x.extend([row_offset + i] * len(ind_slice))
        y.extend(ind_slice)
        vals.extend(val_slice)

    tiledb_fp[x, y] = vals


def optimize_tiledb_array(tiledb_array_uri: str, verbose: bool = True):
    """Consolidate TileDB fragments."""
    if verbose:
        print(f"Optimizing {tiledb_array_uri}")

    frags = tiledb.array_fragments(tiledb_array_uri)
    if verbose:
        print("Fragments before consolidation: {}".format(len(frags)))

    cfg = tiledb.Config()
    cfg["sm.consolidation.step_min_frags"] = 1
    cfg["sm.consolidation.step_max_frags"] = 200
    tiledb.consolidate(tiledb_array_uri, config=cfg)
    tiledb.vacuum(tiledb_array_uri)

    frags = tiledb.array_fragments(tiledb_array_uri)
    if verbose:
        print("Fragments after consolidation: {}".format(len(frags)))


def create_group(output_path, group_name):
    tiledb.group_create(f"{output_path}/{group_name}")
