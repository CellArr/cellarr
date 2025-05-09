import json
import sys
from pathlib import Path

from cellarr.build.buildutils_tiledb_array import optimize_tiledb_array

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def finalize_matrix(args_json: str):
    """Finalize the matrix after all array jobs complete."""
    args = json.loads(args_json)

    # Verify all tasks completed
    completed_dir = Path(args["temp_dir"]) / "completed"
    expected_tasks = len(args["files"])
    completed_tasks = len(list(completed_dir.glob("task_*.json")))

    if completed_tasks != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} tasks but only {completed_tasks} completed")

    # Optimize the TileDB array
    matrix_uri = f"{args['output_dir']}/assays/{args['matrix_options']['matrix_name']}"
    optimize_tiledb_array(matrix_uri)

    # Save completion metadata
    with open(f"{args['temp_dir']}/matrix_metadata.json", "w") as f:
        json.dump(
            {
                "matrix_name": args["matrix_options"]["matrix_name"],
                "files_processed": len(args["files"]),
                "uri": matrix_uri,
            },
            f,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python finalize_matrix.py '<json_args>'")
        sys.exit(1)

    finalize_matrix(sys.argv[1])
