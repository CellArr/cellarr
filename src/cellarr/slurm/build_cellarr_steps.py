import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cellarr.buildutils_tiledb_array import create_tiledb_array

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class SlurmBuilder:
    """SLURM-based builder for CellArrDataset."""

    def __init__(
        self,
        output_dir: str,
        log_dir: str,
        temp_dir: str,
        memory_gb: int = 64,
        time_hours: int = 24,
        cpus_per_task: int = 4,
    ):
        """Initialize the SLURM builder.

        Args:
            output_dir:
                Path to final output directory.

            log_dir:
                Path to store SLURM logs.

            temp_dir:
                Path for temporary files.

            memory_gb:
                Memory per job in GB.

            time_hours:
                Time limit per job in hours.

            cpus_per_task:
                CPUs per task.
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.temp_dir = Path(temp_dir)
        self.memory_gb = memory_gb
        self.time_hours = time_hours
        self.cpus_per_task = cpus_per_task

    def create_slurm_script(
        self,
        job_name: str,
        python_script: str,
        args: Dict,
        dependencies: Optional[str] = None,
        python_env: str = "",
        sbatch_extra_args: str = "",
    ) -> str:
        """Create a SLURM job submission script."""
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
#SBATCH --time={self.time_hours}:00:00
#SBATCH --mem={self.memory_gb}G
#SBATCH --cpus-per-task={self.cpus_per_task}
{sbatch_extra_args}
"""
        if dependencies:
            script += f"#SBATCH --dependency={dependencies}\n"

        script += f"""
{python_env}

python {python_script} '{json.dumps(args)}'
"""
        script_path = self.log_dir / f"{job_name}_submit.sh"
        with open(script_path, "w") as f:
            f.write(script)
        return script_path

    def create_array_script(
        self,
        job_name: str,
        python_script: str,
        args: Dict,
        n_tasks: int,
        dependencies: Optional[str] = None,
        python_env: str = "",
        sbatch_extra_args: str = "",
    ) -> str:
        """Create a SLURM array job submission script."""
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.log_dir}/{job_name}_%A_%a.out
#SBATCH --error={self.log_dir}/{job_name}_%A_%a.err
#SBATCH --time={self.time_hours}:00:00
#SBATCH --mem={self.memory_gb}G
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --array=0-{n_tasks-1}
{sbatch_extra_args}
"""
        if dependencies:
            script += f"#SBATCH --dependency={dependencies}\n"

        script += f"""
{python_env}

python {python_script} '{json.dumps(args)}'
"""
        script_path = self.log_dir / f"{job_name}_array.sh"
        with open(script_path, "w") as f:
            f.write(script)
        return script_path

    def submit_job(self, script_path: str) -> str:
        """Submit a SLURM job and return job ID."""
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        return result.stdout.strip().split()[-1]

    def submit_gene_annotation_job(
        self, files: List[str], gene_options: Dict, python_env: str, sbatch_extra_args: str
    ) -> str:
        """Submit gene annotation processing job."""
        args = {
            "files": files,
            "output_dir": str(self.output_dir),
            "gene_options": gene_options,
            "temp_dir": str(self.temp_dir / "gene_annotation"),
        }

        parent_dir = str(Path(__file__).parent)

        script_path = self.create_slurm_script(
            job_name="cellarr_gene_annot",
            python_script=f"{parent_dir}/process_gene_annotation.py",
            args=args,
            python_env=python_env,
            sbatch_extra_args=sbatch_extra_args,
        )
        return self.submit_job(script_path)

    def submit_sample_metadata_job(
        self, files: List[str], sample_options: Dict, dependency: str, python_env: str, sbatch_extra_args: str
    ) -> str:
        """Submit sample metadata processing job."""
        args = {
            "files": files,
            "output_dir": str(self.output_dir),
            "sample_options": sample_options,
            "temp_dir": str(self.temp_dir / "sample_metadata"),
        }

        parent_dir = str(Path(__file__).parent)

        script_path = self.create_slurm_script(
            job_name="cellarr_sample_meta",
            python_script=f"{parent_dir}/process_sample_metadata.py",
            args=args,
            python_env=python_env,
            sbatch_extra_args=sbatch_extra_args,
            # dependencies=f"afterok:{dependency}",
        )
        return self.submit_job(script_path)

    def submit_cell_metadata_job(
        self, files: List[str], cell_options: Dict, dependency: str, python_env: str, sbatch_extra_args: str
    ) -> str:
        """Submit cell metadata processing job."""
        args = {
            "files": files,
            "output_dir": str(self.output_dir),
            "cell_options": cell_options,
            "temp_dir": str(self.temp_dir / "cell_metadata"),
        }

        parent_dir = str(Path(__file__).parent)

        script_path = self.create_slurm_script(
            job_name="cellarr_cell_meta",
            python_script=f"{parent_dir}/process_cell_metadata.py",
            args=args,
            python_env=python_env,
            sbatch_extra_args=sbatch_extra_args,
            # dependencies=f"afterok:{dependency}",
        )
        return self.submit_job(script_path)

    def submit_matrix_processing(
        self, files: List[str], matrix_options: Dict, dependency: str, python_env: str, sbatch_extra_args: str
    ) -> Tuple[str, str]:
        """Submit matrix processing as SLURM array job."""

        # Create matrix TileDB array
        matrix_uri = str(self.output_dir / "assays" / matrix_options["matrix_name"])
        create_tiledb_array(
            matrix_uri,
            matrix_attr_name=matrix_options.get("matrix_attr_name", "data"),
            matrix_dim_dtype=np.dtype(matrix_options.get("dtype", "float32")),
        )

        # Prepare array job arguments
        array_args = {
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir / f"matrix_{matrix_options['matrix_name']}"),
            "matrix_options": matrix_options,
            "gene_annotation_file": str(self.temp_dir / "gene_annotation/gene_set.json"),
            "files": files,
        }

        parent_dir = str(Path(__file__).parent)

        # Submit array job
        array_script = self.create_array_script(
            job_name=f"matrix_{matrix_options['matrix_name']}",
            python_script=f"{parent_dir}/process_matrix.py",
            args=array_args,
            n_tasks=len(files),
            dependencies=f"afterok:{dependency}",
            python_env=python_env,
            sbatch_extra_args=sbatch_extra_args,
        )
        array_job_id = self.submit_job(array_script)

        # Submit finalization job
        final_args = {
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir / f"matrix_{matrix_options['matrix_name']}"),
            "matrix_options": matrix_options,
            "files": files,
        }

        final_script = self.create_slurm_script(
            job_name=f"finalize_matrix_{matrix_options['matrix_name']}",
            python_script=f"{parent_dir}/finalize_matrix.py",
            args=final_args,
            dependencies=f"afterok:{array_job_id}",
            python_env=python_env,
            sbatch_extra_args=sbatch_extra_args,
        )
        final_job_id = self.submit_job(final_script)

        return array_job_id, final_job_id

    def submit_final_assembly(
        self, matrix_names: List[str], dependencies: List[str], python_env: str, sbatch_extra_args: str
    ) -> str:
        """Submit final assembly job."""
        args = {"input_dir": str(self.output_dir), "output_dir": str(self.output_dir), "matrix_names": matrix_names}

        parent_dir = str(Path(__file__).parent)

        script_path = self.create_slurm_script(
            job_name="cellarr_final_assembly",
            python_script=f"{parent_dir}/final_assembly.py",
            args=args,
            dependencies=f"afterok:{','.join(dependencies)}",
            python_env=python_env,
            sbatch_extra_args=sbatch_extra_args,
        )
        return self.submit_job(script_path)


def main():
    parser = argparse.ArgumentParser(description="Build CellArrDataset using SLURM steps")
    parser.add_argument("--input-manifest", required=True, help="Path to JSON manifest file")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--memory-per-job", type=int, default=64, help="Memory in GB per job")
    parser.add_argument("--time-per-job", type=int, default=24, help="Time in hours per job")
    parser.add_argument("--cpus-per-task", type=int, default=4, help="CPUs per task")
    args = parser.parse_args()

    # Create directories
    base_dir = Path(args.output_dir)
    log_dir = base_dir / "logs"
    temp_dir = base_dir / "temp"
    final_dir = base_dir / "final"
    assays_dir = base_dir / "final/assays"

    for d in [log_dir, temp_dir, final_dir, assays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Read manifest
    with open(args.input_manifest) as f:
        manifest = json.load(f)

    # Initialize builder
    builder = SlurmBuilder(
        output_dir=str(final_dir),
        log_dir=str(log_dir),
        temp_dir=str(temp_dir),
        memory_gb=args.memory_per_job,
        time_hours=args.time_per_job,
        cpus_per_task=args.cpus_per_task,
    )

    # Submit jobs
    gene_job_id = builder.submit_gene_annotation_job(
        manifest["files"], manifest.get("gene_options", {}), manifest["python_env"], manifest["sbatch_extra_args"]
    )

    sample_job_id = builder.submit_sample_metadata_job(
        manifest["files"],
        manifest.get("sample_options", {}),
        gene_job_id,
        manifest["python_env"],
        manifest["sbatch_extra_args"],
    )

    cell_job_id = builder.submit_cell_metadata_job(
        manifest["files"],
        manifest.get("cell_options", {}),
        sample_job_id,
        manifest["python_env"],
        manifest["sbatch_extra_args"],
    )

    # Process matrices
    matrix_options = manifest.get("matrix_options", [{"matrix_name": "counts"}])
    if not isinstance(matrix_options, list):
        matrix_options = [matrix_options]

    matrix_job_ids = []
    for matrix_opt in matrix_options:
        _, final_id = builder.submit_matrix_processing(
            manifest["files"],
            matrix_opt,
            f"{cell_job_id},{gene_job_id}",
            manifest["python_env"],
            manifest["sbatch_extra_args"],
        )
        matrix_job_ids.append(final_id)

    # Submit final assembly
    builder.submit_final_assembly(
        [opt["matrix_name"] for opt in matrix_options],
        matrix_job_ids,
        manifest["python_env"],
        manifest["sbatch_extra_args"],
    )


if __name__ == "__main__":
    main()
