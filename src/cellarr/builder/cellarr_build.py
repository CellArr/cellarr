from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd

from ..build_options import (
    CellMetadataOptions,
    GeneAnnotationOptions,
    MatrixOptions,
    SampleMetadataOptions,
)
from ..CellArrDataset import CellArrDataset
from . import build_cellarrdataset


@dataclass
class BuilderConfig:
    """Configuration for the CellArrDataset builder.

    Provides simplified options for building a CellArrDataset.
    """

    output_path: str
    optimize_tiledb: bool = True
    num_threads: int = 1

    # Matrix options
    matrix_name: str = "counts"
    matrix_dtype: np.dtype = np.float32

    # Gene options
    feature_column: str = "index"
    gene_id_type: np.dtype = np.uint32

    # Cell/Sample options
    cell_id_type: np.dtype = np.uint32
    sample_id_type: np.dtype = np.uint32


class CellArrDatasetBuilder:
    """A builder class to simplify creating CellArrDatasets.

    Example:
        >>> builder = CellArrDatasetBuilder(
        ...     output_path="path/to/output"
        ... )
        >>> builder.add_data(
        ...     adata1,
        ...     "sample1",
        ... )
        >>> builder.add_data(
        ...     adata2,
        ...     "sample2",
        ... )
        >>> dataset = (
        ...     builder.build()
        ... )
    """

    def __init__(self, config: Union[BuilderConfig, dict, str]):
        """Initialize the builder with configuration.

        Args:
            config:
                Either a BuilderConfig object, a dict with config parameters,
                or a path to the output directory (simplest case)
        """
        if isinstance(config, str):
            self.config = BuilderConfig(output_path=config)
        elif isinstance(config, dict):
            self.config = BuilderConfig(**config)
        else:
            self.config = config

        self.data_objects = []
        self.sample_metadata = {}
        self.cell_metadata = {}
        self.gene_metadata = None

        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)

    def add_data(
        self,
        data: Union[str, anndata.AnnData],
        sample_name: str,
        sample_metadata: Optional[Dict] = None,
        cell_metadata: Optional[pd.DataFrame] = None,
    ) -> "CellArrDatasetBuilder":
        """Add a data object to the dataset.

        Args:
            data:
                AnnData object or path to h5ad file

            sample_name:
                Name for this sample

            sample_metadata:
                Optional dictionary of metadata for this sample

            cell_metadata:
                Optional DataFrame with cell metadata

        Returns:
            self for method chaining
        """
        self.data_objects.append((data, sample_name))

        if sample_metadata:
            self.sample_metadata[sample_name] = sample_metadata

        if cell_metadata is not None:
            self.cell_metadata[sample_name] = cell_metadata

        return self

    def set_gene_metadata(self, gene_metadata: Union[pd.DataFrame, List[str]]) -> "CellArrDatasetBuilder":
        """Set gene/feature metadata or list.

        Args:
            gene_metadata:
                DataFrame with gene annotations or list of gene IDs

        Returns:
            self for method chaining
        """
        self.gene_metadata = gene_metadata
        return self

    def build(self) -> CellArrDataset:
        """Build and return the CellArrDataset.

        Returns:
            Constructed CellArrDataset object

        Raises:
            ValueError:
                If no data has been added
        """
        if not self.data_objects:
            raise ValueError("No data objects have been added to build")

        # Prepare options
        matrix_options = MatrixOptions(matrix_name=self.config.matrix_name, dtype=self.config.matrix_dtype)

        gene_options = GeneAnnotationOptions(feature_column=self.config.feature_column, dtype=self.config.gene_id_type)

        cell_options = CellMetadataOptions(dtype=self.config.cell_id_type)

        sample_options = SampleMetadataOptions(dtype=self.config.sample_id_type)

        # Prepare sample metadata
        if self.sample_metadata:
            sample_df = pd.DataFrame.from_dict(self.sample_metadata, orient="index").reset_index(
                names=["cellarr_sample"]
            )
        else:
            sample_df = None

        # Prepare cell metadata
        if self.cell_metadata:
            cell_df = pd.concat(
                [df.assign(cellarr_sample=sample) for sample, df in self.cell_metadata.items()], ignore_index=True
            )
        else:
            cell_df = None

        # Build dataset
        dataset = build_cellarrdataset(
            files=[obj for obj, _ in self.data_objects],
            output_path=self.config.output_path,
            gene_annotation=self.gene_metadata,
            sample_metadata=sample_df,
            cell_metadata=cell_df,
            matrix_options=matrix_options,
            gene_annotation_options=gene_options,
            cell_metadata_options=cell_options,
            sample_metadata_options=sample_options,
            optimize_tiledb=self.config.optimize_tiledb,
            num_threads=self.config.num_threads,
        )

        return dataset
