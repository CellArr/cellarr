import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
import tiledb

__author__ = "Jayaram Kancherla, Tony Kuo"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def create_tiledb_frame_from_chunk(tiledb_uri_path: str, chunk: pd.DataFrame, column_types: Dict[str, np.dtype]):
    """Create a TileDB file from the DataFrame chunk, to persistent storage. This is used by the importer for large
    datasets stored in csv.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the metadata TileDB file.

        chunk:
            Pandas data frame.

        column_types:
            Dictionary specifying the column types for each
            column in the frame.
    """
    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    tiledb.from_pandas(
        tiledb_uri_path,
        chunk,
        mode="schema_only",
        full_domain=True,
        column_types=column_types,
    )


def create_tiledb_frame_from_column_names(
    tiledb_uri_path: str, column_names: List[str], column_types: Dict[str, np.dtype]
):
    """Create a TileDB file with the provided attributes to persistent storage.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the metadata TileDB file.

        column_names:
            Column names of the data frame.

        column_types:
            Dictionary specifying the column types for each
            column in the frame.
    """
    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    df = pd.DataFrame(columns=list(column_names), dtype=column_types)
    for c in df.columns:
        df.loc[0, c] = "None"

    tiledb.from_pandas(
        tiledb_uri_path,
        df,
        mode="schema_only",
        full_domain=True,
        column_types=column_types,
    )


def create_tiledb_frame_from_dataframe(tiledb_uri_path: str, frame: List[str], column_types: dict = None):
    """Create a TileDB file with the provided attributes to persistent storage.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the metadata TileDB file.

        column_names:
            Column names of the data frame.

        column_types:
            Dictionary specifying the column types for each
            column in the frame.
    """
    if os.path.exists(tiledb_uri_path):
        shutil.rmtree(tiledb_uri_path)

    tiledb.from_pandas(tiledb_uri_path, dataframe=frame, column_types=column_types)


def append_to_tiledb_frame(tiledb_uri_path: str, frame: pd.DataFrame, row_offset: int = 0):
    """Create a TileDB file with the provided attributes to persistent storage.

    This will materialize the array directory and all
    related schema files.

    Args:
        tiledb_uri_path:
            Path to create the metadata TileDB file.

        frame:
            Pandas Dataframe to append to TileDB.

        row_offset:
            Row offset to append new rows to.
            Defaults to 0.
    """
    tiledb.from_pandas(tiledb_uri_path, dataframe=frame, mode="append", row_start_idx=row_offset)


# TODO: At some point, hopefully figure out an easy way to identify
# individual column types.
def infer_column_types(frame: pd.DataFrame, col_types: dict) -> Dict[str, str]:
    """Infer column types based on pandas types for each column.

    Note: Currently sets all columns to 'ascii'.

    Args:
        frame:
            DataFrame to infer column types from.

    Returns:
        Dictionary containing column names as keys and
        value representing the column types.
    """
    _to_return_col_types = {}

    if col_types is None:
        col_types = {}

    for col in frame.columns:
        if col in col_types:
            _to_return_col_types[col] = col_types[col]
        else:
            _to_return_col_types[col] = "ascii"

    return _to_return_col_types
