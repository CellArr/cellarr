For simple builds,

# Simple usage
builder = CellArrDatasetBuilder("path/to/output")
builder.add_data(adata1, "sample1")
builder.add_data(adata2, "sample2")
dataset = builder.build()

# With more options
config = BuilderConfig(
    output_path="path/to/output",
    matrix_name="normalized_counts",
    matrix_dtype=np.float32,
    num_threads=4
)

builder = CellArrDatasetBuilder(config)

# Add data with metadata
builder.add_data(
    adata1,
    "sample1",
    sample_metadata={
        "condition": "treatment",
        "batch": "1"
    },
    cell_metadata=cell_meta_df1
)

# Set gene metadata
builder.set_gene_metadata(gene_annotations_df)

# Build the dataset
dataset = builder.build()
