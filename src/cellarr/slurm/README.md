
# manifest json

```json
{
    "files": [
        "/path/to/dataset1.h5ad",
        "/path/to/dataset2.h5ad"
    ],
    "matrix_options": [
        {
            "matrix_name": "counts",
            "dtype": "uint32"
        },
        {
            "matrix_name": "normalized",
            "dtype": "float32"
        }
    ],
    "gene_options": {
        "feature_column": "index"
    },
    "sample_options": {
        "metadata": {
            "sample_1": {
                "condition": "control",
                "batch": "1"
            },
            "sample_2": {
                "condition": "treatment",
                "batch": "1"
            }
        }
    },
    "cell_options": {
        "column_types": {
            "cell_type": "ascii",
            "quality_score": "float32"
        },
    },
    "python_env": """
. /system/gredit/clientos/etc/profile

ml Miniforge3
conda activate biocpy_miniforge

~/.conda/envs/biocpy_miniforge/bin/python --version
which python
python --version
""",
    "sbatch_extra_args": """
#SBATCH --qos=short
#SBATCH --partition=defq
#SBATCH -t 0-02:00
"""
}
```


Run

```sh

python build_cellarr_steps.py \
    --input-manifest manifest.json \
    --output-dir /path/to/output \
    --memory-per-job 64 \
    --cpus-per-task 4

```
