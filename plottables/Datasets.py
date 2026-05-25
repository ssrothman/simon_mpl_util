from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds

import numpy as np
import awkward as ak

import hist
import matplotlib.axes


from simonplot.util.comparison import ComparisonHistStruct
from simonpy.AbitraryBinning import ArbitraryBinning

from typing import List, Union, override

from .DatasetBase import SingleDatasetBase, DatasetStackBase, DatasetComparisonBase
from simonplot.typing.Protocols import BaseDatasetProtocol
import pyarrow._compute as pc2

class DatasetStack(DatasetStackBase):
    def __init__(self, key : str, color : str | None, label : str, datasets : list[BaseDatasetProtocol], showstack : bool = True):
        self._key = key
        self._color = color
        self._label = label
        self._datasets = datasets
        self._showStack = showstack

    def streaming_fill_histogram(self, H : hist.Hist, 
                                 variables : dict[str, ds.Expression], 
                                 weight : ds.Expression | None,
                                 mask : ds.Expression | None,
                                 batch_size : int = 1 << 20,
                                 batch_readahead : int = 4,
                                 fragment_readahead : int = 4,
                                 use_threads : bool = True):
        import pyarrow.compute as pc
        # Specialized method to fill a histogram in a streaming way, without loading the entire dataset into memory.
        # This is useful for very large datasets that cannot fit into memory.
        for dataset in self._datasets:
            assert isinstance(dataset, DatasetStack) or isinstance(dataset, ParquetDataset), "Currently only ParquetDataset and DatasetStack are supported in DatasetStack for streaming_fill_histogram"
            dataset.streaming_fill_histogram(
                H, 
                variables, weight, mask, 
                batch_size, 
                batch_readahead, fragment_readahead, 
                use_threads
            )
        
class DatasetComparison(DatasetComparisonBase):
    def __init__(self, key : str, color : str | None, label : str, ylabel : str, dataset1 : BaseDatasetProtocol, dataset2 : BaseDatasetProtocol, kind : ComparisonHistStruct._SUPPORTED_MODES):
        self._key = key
        self._color = color
        self._label = label
        self._ylabel = ylabel
        self._dataset1 = dataset1
        self._dataset2 = dataset2
        self._kind = kind

    @property
    def ylabel(self):
        return self._ylabel
    
class NanoEventsDataset(SingleDatasetBase):
    def __init__(self, key : str, color : str | None, label : str, fname, **options):
        self._key = key
        self._color = color
        self._label = label

        #suppress warnings
        NanoAODSchema.warn_missing_crossrefs = False

        import coffea
        version = coffea._version.version_tuple
        if int(version[0]) >= 2025 and int(version[1]) >= 11 and int(version[2]) >= 0:
            options['mode'] = 'virtual'
        else:
            options['delayed'] = False

        self._events = NanoEventsFactory.from_root(
            fname,
            **options 
        ).events()
        
    def ensure_columns(self, columns):
        # NanoEvents loads all columns on demand, so nothing to do here
        pass

    def get_column(self, column_name, collection_name=None):
        if '.' in column_name:
            raise ValueError("NanoEventsDataset.get_column: column_name '%s' contains '.'! Instead use collection_name argument."%(column_name))
        
        if collection_name is not None:
            return ak.materialize(self._events[collection_name][column_name])
        else:
            return ak.materialize(self._events[column_name])
        
    @property
    def num_rows(self):
        return len(self._events)
    
class ParquetDataset(SingleDatasetBase):
    def __init__(self, key : str, color : str | None, label : str, path, filesystem=None):
        self._key = key
        self._color = color
        self._label = label

        self._dataset = ds.dataset(path, format="parquet", filesystem=filesystem)
            
    def ensure_columns(self, columns):
        has_everything = True
        if hasattr(self, '_table'):
            for col in columns:
                if col not in self._table.column_names:
                    has_everything = False
                    break
        else:
            has_everything = False

        if not has_everything:
            self._table = self._dataset.to_table(columns=columns)
    
    def get_column(self, column_name, collection_name=None):
        if collection_name is not None:
            raise NotImplementedError("ParquetDataset does not support collection_name argument")
        
        if not hasattr(self, '_table'):
            raise RuntimeError("ParquetDataset.ensure_columns must be called before get_column")
        
        if column_name not in self._table.column_names:
            raise RuntimeError("Column %s not loaded! Call ensure_columns() first"%column_name)

        return self._table[column_name].to_numpy()
    
    @property
    def num_rows(self):
        if hasattr(self, '_table'):
            return self._table.num_rows
        else:
            return self._dataset.count_rows()
    
    #extra properties for parquetdatasets for utility
    @property
    def files(self):
        return self._dataset.files
    
    @property 
    def filesystem(self):
        return self._dataset.filesystem
    
    @property
    def schema(self):
        return self._dataset.schema

    def streaming_fill_histogram(self, H : hist.Hist, 
                                 variables : dict[str, ds.Expression], 
                                 weight : ds.Expression | None,
                                 mask : ds.Expression | None,
                                 batch_size : int = 1 << 20,
                                 batch_readahead : int = 1,
                                 fragment_readahead : int = 1,
                                 use_threads : bool = True):
        # Specialized method to fill a histogram in a streaming way, without loading the entire dataset into memory.
        # This is useful for very large datasets that cannot fit into memory.

        # Create an iterator over the dataset in batches
        columns = variables
        if weight is not None:
            import pyarrow.compute as pc
            columns['weight'] = pc.multiply(weight, self._weight)
            
        iterator = self._dataset.to_batches(
            columns = columns,
            filter = mask,
            batch_size = batch_size,
            batch_readahead = batch_readahead,
            fragment_readahead = fragment_readahead,
            use_threads = use_threads,
        )
        from tqdm import tqdm
        iterator = tqdm(iterator, desc='Filling histogram', unit='batch')
        for batch in iterator:
            # Evaluate the variables and weights for the current batch
            H.fill(**{name : value for name, value in zip(batch.column_names, batch.columns)})