import json
import sys
from pathlib import Path

import pandas as pd

from cellarr.utils import utils_anndata as uad
from cellarr.build.buildutils_tiledb_frame import create_tiledb_frame_from_dataframe

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def process_sample_metadata(args_json: str):
    """Process and create sample metadata store.

    Creates sample metadata including:
    - Basic sample information
    - Cell counts per sample
    - Original gene sets
    - Sample index information
    """
    args = json.loads(args_json)

    # Extract information from files
    files_cache = uad.extract_anndata_info(args["files"], var_feature_column=args.get("feature_column", "index"))

    # Get cell counts for each sample
    cell_counts = uad.scan_for_cellcounts(files_cache)

    # Create basic sample metadata
    sample_names = [f"sample_{idx + 1}" for idx in range(len(args["files"]))]
    sample_metadata = pd.DataFrame(
        {"cellarr_sample": sample_names, "cellarr_cell_counts": cell_counts, "cellarr_filename": args["files"]}
    )

    # Add sample indices for efficient slicing
    counter = sample_metadata["cellarr_cell_counts"].shift(1)
    counter.iloc[0] = 0
    sample_metadata["cellarr_sample_start_index"] = counter.cumsum().astype(int)

    # Calculate end indices
    ends = sample_metadata["cellarr_sample_start_index"].shift(-1)
    ends.iloc[-1] = int(sample_metadata["cellarr_cell_counts"].sum())
    ends = ends - 1
    sample_metadata["cellarr_sample_end_index"] = ends.astype(int)

    # Add original gene sets for each sample
    gene_sets = uad.scan_for_features(files_cache, unique=False)
    sample_metadata["cellarr_original_gene_set"] = [",".join(genes) for genes in gene_sets]

    # Add any custom metadata if provided in options
    custom_metadata = args.get("sample_options", {}).get("metadata", {})
    for sample, metadata in custom_metadata.items():
        for key, value in metadata.items():
            if key not in sample_metadata.columns:
                sample_metadata[key] = None
            sample_idx = sample_metadata.index[sample_metadata["cellarr_sample"] == sample][0]
            sample_metadata.at[sample_idx, key] = value

    # Create TileDB store
    create_tiledb_frame_from_dataframe(f"{args['output_dir']}/sample_metadata", sample_metadata)

    # Save metadata for subsequent steps
    metadata = {"num_samples": len(sample_names), "total_cells": int(sample_metadata["cellarr_cell_counts"].sum())}

    Path(args["temp_dir"]).mkdir(parents=True, exist_ok=True)
    with open(f"{args['temp_dir']}/sample_metadata.json", "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_sample_metadata.py '<json_args>'")
        sys.exit(1)

    process_sample_metadata(sys.argv[1])
