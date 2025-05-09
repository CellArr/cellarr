import json
import sys

import pandas as pd

from cellarr.utils import utils_anndata as uad
from cellarr.build.buildutils_tiledb_frame import (
    create_tiledb_frame_from_dataframe,
)

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def process_cell_metadata(args_json: str):
    """Process and create cell metadata store.

    Creates cell metadata including:
    - Sample mapping
    - Cell indices within samples
    - Original cell annotations from input files
    """
    args = json.loads(args_json)

    # Extract cell metadata with specific column subset if provided
    cell_meta_columns = args.get("cell_options", {}).get("column_types", {})
    files_cache = uad.extract_anndata_info(
        args["files"], obs_subset_columns=list(cell_meta_columns.keys()) if cell_meta_columns else None
    )

    # Get cell counts
    cell_counts = uad.scan_for_cellcounts(files_cache)

    # Create sample mapping for each cell
    sample_per_cell = []
    cell_index_in_sample = []
    sample_names = [f"sample_{idx+1}" for idx in range(len(args["files"]))]

    for idx, count in enumerate(cell_counts):
        sample_per_cell.extend([sample_names[idx]] * count)
        cell_index_in_sample.extend(range(count))

    # Create base cell metadata
    cell_metadata = pd.DataFrame(
        {"cellarr_sample": sample_per_cell, "cellarr_cell_index_in_sample": cell_index_in_sample}
    )

    # Add original cell annotations from input files
    if cell_meta_columns:
        original_meta = uad.scan_for_cellmetadata(files_cache)
        if not original_meta.empty:
            # Ensure index alignment
            original_meta.reset_index(drop=True, inplace=True)
            cell_metadata = pd.concat([cell_metadata, original_meta], axis=1)

    # Create TileDB store
    create_tiledb_frame_from_dataframe(
        f"{args['output_dir']}/cell_metadata", cell_metadata, column_types=cell_meta_columns
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_cell_metadata.py '<json_args>'")
        sys.exit(1)

    process_cell_metadata(sys.argv[1])
