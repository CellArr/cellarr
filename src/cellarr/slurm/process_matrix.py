import json
import os
import sys
from pathlib import Path

import tiledb

from cellarr.utils import utils_anndata as uad
from cellarr.build.buildutils_tiledb_array import write_csr_matrix_to_tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def process_matrix_file(args_json: str):
    """Process a single file for matrix creation."""
    args = json.loads(args_json)

    # Get SLURM array task ID
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # Load gene set mapping
    with open(args["gene_annotation_file"]) as f:
        gene_set = json.load(f)
    gene_map = {gene: idx for idx, gene in enumerate(gene_set)}

    # Get file and offset for this task
    file_info = args["files"][task_id]
    input_file = file_info

    # get sample offset
    sample_uri = tiledb.open(f"{args['output_dir']}/sample_metadata", "r")
    sample_row = sample_uri.df[task_id]
    row_offset = sample_row["cellarr_sample_start_index"]

    # Process the file
    matrix = uad.remap_anndata(input_file, gene_map, layer_matrix_name=args["matrix_options"]["matrix_name"])

    # Write to TileDB
    matrix_uri = f"{args['output_dir']}/assays/{args['matrix_options']['matrix_name']}"
    write_csr_matrix_to_tiledb(matrix_uri, matrix[args["matrix_options"]["matrix_name"]], row_offset=row_offset)

    # Save task completion marker
    Path(args["temp_dir"]).mkdir(parents=True, exist_ok=True)
    Path(args["temp_dir"] + "/completed").mkdir(parents=True, exist_ok=True)
    task_marker = Path(args["temp_dir"]) / "completed" / f"task_{task_id}.json"
    task_marker.parent.mkdir(exist_ok=True)
    with open(task_marker, "w") as f:
        json.dump({"file": input_file, "cells_processed": matrix[args["matrix_options"]["matrix_name"]].shape[0]}, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_matrix.py '<json_args>'")
        sys.exit(1)

    process_matrix_file(sys.argv[1])
