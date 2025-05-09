import re
from functools import lru_cache
from typing import List, Union
from warnings import warn

import numpy as np
import pandas as pd
import tiledb
from scipy import sparse as sp

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@lru_cache
def get_schema_names_frame(tiledb_obj: tiledb.Array) -> List[str]:
    """Get Attributes from a TileDB object.

    Args:
        tiledb_obj:
            A TileDB object.

    Returns:
        List of schema attributes.
    """
    columns = []
    for i in range(tiledb_obj.schema.nattr):
        columns.append(tiledb_obj.schema.attr(i).name)

    return columns


def subset_frame(
    tiledb_obj: tiledb.Array,
    subset: Union[slice, str],
    columns: list,
    primary_key_column_name: str = None,
) -> pd.DataFrame:
    """Subset a TileDB object.

    Args:
        tiledb_obj:
            TileDB object to subset.

        subset:
            A :py:class:`slice` to subset.

            Alternatively, may also provide a TileDB query expression.

        columns:
            List specifying the atrributes from the schema to extract.

        primary_key_column_name:
            The primary key to filter for matches when a
            :py:class:`~tiledb.QueryCondition` is used.

    Returns:
        A sliced `DataFrame` with the subset.
    """

    if isinstance(subset, str):
        warn(
            "provided subset is string, its expected to be a valid tiledb expression",
            UserWarning,
        )

        if primary_key_column_name is None:
            raise ValueError("'primary_key_column_name' cannot be 'None'.")

        if columns is None:
            all_columns = []
        else:
            all_columns = columns.copy()

        all_columns.append(primary_key_column_name)
        query = tiledb_obj.query(cond=subset, attrs=list(set(all_columns)))
        mask = tiledb_obj.attr(primary_key_column_name).fill
        if isinstance(mask, bytes):
            mask = mask.decode("ascii")
        data = query.df[:][primary_key_column_name]
        filtered = np.where(data != mask)[0]
        data = tiledb_obj.df[filtered]
    else:
        data = tiledb_obj.df[subset][columns]

    re_null = re.compile(pattern="\x00")  # replace null strings with nan
    result = data.replace(regex=re_null, value=np.nan)

    return result


def _remap_index(indices: List[int]) -> List[int]:
    _map = {}
    _new_indices = []

    for ridx, r in enumerate(list(sorted(set(indices)))):
        _map[r] = ridx

    for r in list(indices):
        _new_indices.append(_map[r])

    return _new_indices, len(_map)


def subset_array(
    tiledb_obj: tiledb.Array,
    row_subset: Union[slice, list, tuple],
    column_subset: Union[slice, list, tuple],
    shape: tuple,
) -> Union[np.ndarray, sp.csr_matrix]:
    """Subset a TileDB storing array data.

    Uses multi_index to slice.

    Args:
        tiledb_obj:
            A TileDB object

        row_subset:
            Subset along the row axis.

        column_subset:
            Subset along the column axis.

        shape:
            Shape of the entire matrix.

    Returns:
        if the TileDB object is sparse, returns a sparse array in a coo format
        otherwise a numpy object.
    """

    if not tiledb_obj.schema.sparse:
        return tiledb_obj[row_subset, column_subset]["data"]

    data = tiledb_obj.multi_index[row_subset, column_subset]

    # Fallback just in case
    shape = (
        tiledb_obj.nonempty_domain()[0][1] + 1,
        tiledb_obj.nonempty_domain()[1][1] + 1,
    )

    mat = sp.coo_matrix(
        (data["data"], (data["cell_index"], data["gene_index"])),
        shape=shape,
    ).tocsr()

    if row_subset is not None and row_subset != slice(None):
        mat = mat[row_subset, :]

    if column_subset is not None and column_subset != slice(None):
        mat = mat[:, column_subset]

    # _cell_rows, _ = _remap_index(data["cell_index"])
    # _gene_cols, _ = _remap_index(data["gene_index"])

    # mat = sp.coo_matrix(
    #     (data["data"], (_cell_rows, _gene_cols)),
    #     shape=shape,
    # )

    return mat


def get_a_column(tiledb_obj: tiledb.Array, column_name: Union[str, List[str]]) -> list:
    """Access column(s) from the TileDB object.

    Args:
        tiledb_obj:
            A TileDB object.

        column_name:
            Name(s) of the column to access.

    Returns:
        List containing the column values.
    """
    if column_name not in get_schema_names_frame(tiledb_obj):
        raise ValueError(f"Column '{column_name}' does not exist.")

    if isinstance(column_name, str):
        column_name = [column_name]

    return tiledb_obj.query(attrs=column_name).df[:]


@lru_cache
def get_index(tiledb_obj: tiledb.Array) -> list:
    """Get the index of the TileDB object.

    Args:
        tiledb_obj:
            A TileDB object.

    Returns:
        A list containing the index values.
    """
    _index = tiledb_obj.unique_dim_values("__tiledb_rows")
    return [x.decode() for x in _index]


def _match_to_list(x: list, query: list):
    return sorted([x.index(q) for q in query])


def _is_list_strings(x: list):
    _ret = False

    if isinstance(x, (list, tuple)) and all(isinstance(y, str) for y in x):
        _ret = True

    return _ret
