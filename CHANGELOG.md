# Changelog

## Version 0.5.6 - 0.5.9

- slrum runner's manifest now accepts an additional key: "sbatch_extra_args", to add any additional SBATCH headers to the generated scripts.
- Added tutorial using cellxgene collection.
- Fixed an issue with slicing when subset arguments are either `None` or `slice(None)`

## Version 0.5.3 - 0.5.5

- 'config'  is renamed to 'context_or_config' to accept both TileDB's context or config objects. This supports in-memory TileDB contexts from [Phil](https://github.com/hanslovsky) [[#79](https://github.com/TileOme/cellarr/pull/79)].
- Fixes the condition for path to assay groups.
- Supports querying of dense arrays with the `CellArrDataset` class.


## Version 0.5.1 - 0.5.2

- Support csc matrices in layers, although not common ran into a situation where the anndata object was stored from R.

## Version 0.5.0

- Construct cellarr TileDB files on HPC environments based on slurm
(reference: [#61](https://github.com/cellarr/cellarr/pull/61))

## Version 0.4.0

- chore: Remove Python 3.8 (EOL).
- precommit: Replace docformatter with ruff's formatter.

## Version 0.3.2

- Functionality to iterate over samples and cells.
- Explicitly mention that slicing defaults to TileB's behavior, inclusive of upper bounds.

## Version 0.3.0 - 0.3.1

This version introduces major improvements to matrix handling, storage, and performance, including support for multiple matrices in H5AD/AnnData workflows and optimizations for ingestion and querying.

**Support for multiple matrices**:
  - Both `build_cellarrdataset` and `CellArrDataset` now support multiple matrices. During ingestion, a TileDB group called `"assays"` is created to store all matrices, along with group-level metadata.

This may introduce breaking changes with the default parameters based on how these classes are used. Previously to build the TileDB files:

```python
dataset = build_cellarrdataset(
    output_path=tempdir,
    files=[adata1, adata2],
    matrix_options=MatrixOptions(matrix_name="counts", dtype=np.int16),
    num_threads=2,
)
```

Now you may provide a list of matrix options for each layers in the files.

```python
dataset = build_cellarrdataset(
    output_path=tempdir,
    files=[adata1, adata2],
    matrix_options=[
        MatrixOptions(matrix_name="counts", dtype=np.int16),
        MatrixOptions(matrix_name="log-norm", dtype=np.float32),
    ],
    num_threads=2,
)
```

Querying follows a similar structure:
```python
cd = CellArrDataset(
    dataset_path=tempdir,
    assay_tiledb_group="assays",
    assay_uri=["counts", "log-norm"]
)
```
`assay_uri` is relative to `assay_tiledb_group`. For backwards compatibility, `assay_tiledb_group` can be an empty string.

**Parallelized ingestion**:
The build process now uses `num_threads` to ingest matrices concurrently. Two new columns in the sample metadata, `cellarr_sample_start_index` and `cellarr_sample_end_index`, track sample offsets, improving matrix processing.
  - Note: The process pool uses the `spawn` method on UNIX systems, which may affect usage on windows machines.

**TileDB query condition fixes**:
Fixed a few issues with fill values represented as bytes (seems to be common when ascii is used as the column type) and in general filtering operations on TileDB Dataframes.

**Index remapping**:
Improved remapping of indices from sliced TileDB arrays for both dense and sparse matrices. This is not a user facing function but an internal slicing operation.

**Get a sample**:
Added a method to access all cells for a particular sample. you can either provide an index or a sample id.

```python
sample_1_slice = cd.get_cells_for_sample(0)
```

Other updates to documentation, tutorials, the README, and additional tests.

## Version 0.2.4 - 0.2.5

- Provide options to extract an expected set of cell metadata columns across datasets.
- Update documentation and tests.

## Version 0.2.1 - 0.2.3

* Implement dunder methods `__len__`,  `__repr__` and `__str__` for the `CellArrDatasetSlice` class
* Add property `shape` to the same class
* Improve package load time


## Version 0.2.0

- Thanks to [@tony-kuo](https://github.com/tony-kuo), the package now includes a built-in dataloader for the pytorch-lightning framework,
for single cells expression profiles, training labels, and study labels. The dataloader uniformly samples across training labels and study labels to create a diverse batch of cells.

- Minor fixes for CSV to TileDB conversion for the `cell_metadata` object.

## Version 0.1.0 - 0.1.3

This is the first release of the package to support both creation and access to large
collection of files based on TileDB.

- Provide a build method to create the TileDB collection from a series of data objects.
- Provides `CellArrDataset` class to query these objects on disk.
- Implements access and coerce methods to interop with other experimental data packages.
