import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import tiledb
from cellarr.queryutils_tiledb_frame import subset_array

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

def create_dense_matrix():
    tempdir = tempfile.mkdtemp()

    d1 = tiledb.Dim(name="cell_index", domain=(0, 3), tile=2, dtype=np.int32)
    d2 = tiledb.Dim(name="gene_index", domain=(0, 3), tile=2, dtype=np.int32)
    dom = tiledb.Domain(d1, d2)
    a = tiledb.Attr(name="data", dtype=np.int32)
    sch = tiledb.ArraySchema(domain=dom, sparse=False, attrs=[a])
    tiledb.Array.create(tempdir, sch)

    data = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int32
    )
    with tiledb.open(tempdir, "w") as A:
        A[:] = data

    return tempdir


def test_query_cellarrdataset():

    array_uri = create_dense_matrix()

    tdb = tiledb.open(array_uri, "r")
    res = subset_array(tdb, row_subset=slice(0,2), column_subset=slice(None), shape=(4,4))

    assert res.shape == (2,4)
    assert np.all(res == np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
