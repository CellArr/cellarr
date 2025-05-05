# CellArr Collections: Using cellxgene datasets

So you've got loads of single-cell data piling up on your hard drive, and making your analysis workflows slower? Welcome to the club!

Cell Arrays is a Python package that provides a TileDB-backed store for large collections of genomic experimental data, such as millions of cells across multiple single-cell experiment objects.

The `CellArrDataset` is designed to store single-cell RNA-seq datasets but can be generalized to store any 2-dimensional experimental data.

## Tutorial: A Real-World Example

To demonstrate how to build your own `CellArr` compatible TileDB collection, we'll download a few datasets from cellxgene.

- Buccal and Labial Atlas (30K cells): https://datasets.cellxgene.cziscience.com/e4e24745-5822-4838-8ad9-41e116b71964.h5ad
- Astrocytes (18K cells): https://datasets.cellxgene.cziscience.com/e2012ca6-059f-4af7-80fe-886579bfaeee.h5ad
- Microglia (1.6K cells): https://datasets.cellxgene.cziscience.com/f9bca543-f82a-4863-bd62-b270b634ea10.h5ad
- Oligodendrocytes (25K cells): https://datasets.cellxgene.cziscience.com/da57b970-3c33-49bb-a951-bfc9b99dbd23.h5ad
- thymus scRNA-seq atlas - B cell subset: https://datasets.cellxgene.cziscience.com/fc6eb410-6603-4e98-9d5f-32c20ffa2456.h5ad

Don't worry if your workflow is different - the beauty of `CellArr` is, its flexibility. The only thing you need to care about are the key column names that make your files CellArr-compatible. Refer to the image in the README for these key column names.

## Step 0: Download All the Things

First, let's grab those datasets.

```python
!wget https://datasets.cellxgene.cziscience.com/fc6eb410-6603-4e98-9d5f-32c20ffa2456.h5ad
!wget https://datasets.cellxgene.cziscience.com/da57b970-3c33-49bb-a951-bfc9b99dbd23.h5ad
!wget https://datasets.cellxgene.cziscience.com/e4e24745-5822-4838-8ad9-41e116b71964.h5ad
!wget https://datasets.cellxgene.cziscience.com/e2012ca6-059f-4af7-80fe-886579bfaeee.h5ad
!wget https://datasets.cellxgene.cziscience.com/f9bca543-f82a-4863-bd62-b270b634ea10.h5ad
```

```python
files = ["fc6eb410-6603-4e98-9d5f-32c20ffa2456.h5ad",
        "da57b970-3c33-49bb-a951-bfc9b99dbd23.h5ad",
        "e4e24745-5822-4838-8ad9-41e116b71964.h5ad",
        "e2012ca6-059f-4af7-80fe-886579bfaeee.h5ad",
        "f9bca543-f82a-4863-bd62-b270b634ea10.h5ad"]
```

## Step 1: Identify Your Feature Space

By default, CellArr's build scans all your files and computes a union of all features. You can customize this if:

1. You already have a pre-filtered gene list
2. You want more control over what features make the final cut

After peeking at these datasets, we notice they use Ensembl IDs in the `var` index. Let's use those as our feature space, which helps align expression vectors across datasets.

```python
def get_feature_space(path):
    ad = anndata.read_h5ad(path, "r")
    return ad.var.index.tolist()
```

```python
from tqdm import tqdm

feature_set = set()

for i in tqdm(files):
    feature_set = feature_set.union(get_feature_space(i))

print("Total # of features:", len(feature_set))
```

> 💡 **Pro Tip:** If you're processing enough files to finish a Netflix series while waiting, consider using joblib or other parallelization tools.

```python
import pandas as pd

sorted_features = sorted(list(feature_set))
FEATURE_DF = pd.DataFrame({"cellarr_gene_index": sorted_features})
FEATURE_DF.head()
```

> 📝 **Note:** The current version of CellArr calls the index column `cellarr_gene_index`. We might generalize this name in a future version.

## Step 2: Wrangling Cell Annotations

By default, CellArr aggregates ALL cell annotations across datasets, which can get... interesting. You have two options:

1. Combine all `obs` objects and then filter out the columns you don't need
2. Provide a list of column names and their types, and CellArr will use "None" for missing values across datasets

We'll go with option 1 since we're working with a manageable number of datasets. But be warned: with large datasets, this might need more RAM.

```python
def get_cell_annotations(path):
    ad = anndata.read_h5ad(path, "r")
    return ad.obs

from tqdm import tqdm

obs = []

for i in tqdm(files):
    obs.append(get_cell_annotations(i))

combined_obs = pd.concat(obs, ignore_index=True, axis=0, sort=False).astype(pd.StringDtype())
combined_cell_df = combined_obs.reset_index(drop=True)

# Let's keep only the columns we actually care about - goodbye, clutter!
columns_to_keep = ["donor_id", "sample_source", "tissue_type",
                  "is_primary_data", "author_cell_type", "cell_type", "disease", "sex", "tissue"]
CELL_ANNOTATIONS_DF = combined_cell_df[columns_to_keep]
CELL_ANNOTATIONS_DF  # Behold, your streamlined metadata!
```

## Step 3a: Matrix Maneuvers (Optional)

> 💡 **Heads up:** This step is optional.

CellArr's build process only recognizes matrices in the `layers` slot of the `AnnData` structure. The files we got from cellxgene use `X` instead, so we need to do a quick switch.

```python
def process_adata(path):
    ad = anndata.read_h5ad(path)
    ad.layers["counts"] = ad.X  # Move the matrix to where CellArr expects it
    return ad

adatas = [process_adata(x) for x in files]  # Apply to all files
```

```python
sum([len(a.obs) for a in adatas])  # Total cell count - prepare to be impressed (or terrified)
```

## Step 3b: Building The TileDB Files

This is where CellArr transforms your collection of disparate datasets into a single, queryable TileDB.

```python
from cellarr import build_cellarrdataset, CellArrDataset, MatrixOptions
import numpy as np
import os
import shutil

# Let's create a fresh directory for our collection
output_dir = "./my_collection.tdb"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.mkdir(output_dir)

# Build the dataset - this is where the magic happens!
dataset = build_cellarrdataset(
    gene_annotation=FEATURE_DF,
    cell_metadata=CELL_ANNOTATIONS_DF,
    output_path="./my_collection.tdb",
    files=adatas,
    matrix_options=MatrixOptions(matrix_name="counts", dtype=np.int16),
    num_threads=4,  # Adjust based on your CPU - more threads = more speed (usually)
)
```

> 🎉 **That's it!** Really, it's that simple.

For those running on a high-performance computing cluster, check out the [slurm-based build process](https://github.com/CellArr/cellarr?tab=readme-ov-file#building-on-hpc-environments-with-slurm).

For more complex workflows, refer to the [full CellArr documentation](https://github.com/CellArr/cellarr).

## Exploring Your New TileDB Collection

Now for the fun part - let's interact with our collection! Having a single collection opens up all sorts of possibilities:

- Explore marker expression across cell types, tissues, and disease states or any available cell metadata
- Develop new methods without data wrangling headaches
- Train machine learning models on unified data

> 🤖 **Did you know?** CellArr includes a PyTorch dataloader to jumpstart your ML training.

```python
from cellarr import CellArrDataset
import tiledb

# Optional: customize TileDB configuration
cfg = tiledb.Config()

output_dir = "./my_collection.tdb"

my_collection = CellArrDataset(output_dir, config_or_context=cfg)
my_collection
```

Get a quick list of all available gene symbols:

```python
gene_symbols = my_collection.get_gene_annotation_index()

# Let's grab a random sample to play with
from random import sample
my_gene_list = sample(gene_symbols, 10)
print("Random genes to investigate:", my_gene_list)
```

### Subsetting: Get Only What You Need

The whole point of CellArr is to efficiently slice and dice your data. Instead of loading the entire count matrix, you can extract just the bits you care about.

Let's grab expression counts for our random gene set in the first 100 cells:

```python
expression_data = my_collection[0:100, my_gene_list]
print(expression_data)
```

The returned `CellArrDatasetSlice` contains everything you need:
1. Matrices (e.g., counts)
2. Cell annotations
3. Feature metadata

Convert this slice to your favorite format - `AnnData` for the Scanpy enthusiasts:

```python
print(expression_data.to_anndata())  # Scanpy-ready!
```

Or `SummarizedExperiment` for the Bioconductor/BiocPy ecosystem:

```python
print(expression_data.to_summarizedexperiment())
```

### Let's Make Some Plots!

Time to put our data to work with some visualizations. First, let's see what metadata we have to play with:

```python
my_collection.get_cell_metadata_columns()
```

Let's focus on specific diseases:

```python
disease_labels = my_collection.get_cell_metadata_column("disease")
disease_labels.value_counts()  # What diseases do we have?
```

Let's zoom in on Alzheimer's disease cells:

```python
import numpy as np
cells_of_interest = np.where(disease_labels == "Alzheimer disease")
print(f"Found {len(cells_of_interest[0])} cells with Alzheimer's disease!")
```

Now, let's get expression data for our random gene list in these Alzheimer's cells:

```python
ad_gene_list = my_gene_list  # Using our random genes from earlier
ad_exprs = my_collection[cells_of_interest, ad_gene_list]
print(ad_exprs)  # Our Alzheimer's disease-specific dataset
```

```python
# Let's check out the metadata for these cells
ad_exprs.cell_metadata.head()
```

Finally, let's visualize gene expression across different cell types:

```python
import pandas as pd
import seaborn as sns

# Combine expression data with metadata
gene_with_meta = pd.concat(
    [
        pd.DataFrame(
            ad_exprs.matrix["counts"].todense(),
            columns=ad_gene_list
        ).reset_index(drop=True),
        ad_exprs.cell_metadata.reset_index(drop=True)
    ],
    axis=1
)

# Compute total expression by cell_type
mean_expression_by_cell_type = gene_with_meta[
    ["cell_type"] + ad_gene_list
].groupby("cell_type").sum()

sns.heatmap(mean_expression_by_cell_type, cmap="crest")
```

Voilà! A beautiful heatmap showing gene expression levels across cell types.

## In Conclusion...

While this was a simple example, it showcases how CellArr lets you:
1. Quickly combine diverse datasets
2. Efficiently query specific subsets of your data
3. Analyze large collections

For more details, check out the [CellArr documentation](https://github.com/CellArr/cellarr) or reach out for help. Happy analyzing!
