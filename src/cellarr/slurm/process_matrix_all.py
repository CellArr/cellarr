import json

import numpy as np

from cellarr.utils import utils_anndata as uad
from cellarr.build.buildutils_tiledb_array import create_tiledb_array, write_csr_matrix_to_tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def process_matrix(args_json: str):
    args = json.loads(args_json)

    # Load gene set
    with open(args["gene_annotation_file"]) as f:
        gene_set = json.load(f)

    # Create gene set mapping
    gene_map = {gene: idx for idx, gene in enumerate(gene_set)}

    # Create TileDB array
    create_tiledb_array(
        f"{args['output_dir']}/assays/{args['matrix_options']['matrix_name']}",
        matrix_dim_dtype=np.dtype(args["matrix_options"].get("dtype", "float32")),
    )

    # Process each file
    offset = 0
    for file in args["files"]:
        matrix = uad.remap_anndata(file, gene_map, layer_matrix_name=args["matrix_options"]["matrix_name"])

        write_csr_matrix_to_tiledb(
            f"{args['output_dir']}/assays/{args['matrix_options']['matrix_name']}",
            matrix[args["matrix_options"]["matrix_name"]],
            row_offset=offset,
        )

        offset += matrix[args["matrix_options"]["matrix_name"]].shape[0]
