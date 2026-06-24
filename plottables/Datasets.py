import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds

import numpy as np
import awkward as ak

import hist
import matplotlib.axes
import pyarrow.compute as pc

from simonplot.util.comparison import ComparisonHistStruct
from simonplot.util.rate import RateHistStruct, RateStruct
from simonplot.variable.Variable import ProfileVariable, RateVariable
from simonpy.AbitraryBinning import ArbitraryBinning

from typing import Any, List, Tuple, Union, override

from .DatasetBase import SingleDatasetBase, DatasetStackBase, DatasetComparisonBase
from simonplot.typing.Protocols import BaseDatasetProtocol, CutProtocol, VariableProtocol

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
        from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

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

    @override
    def get_range(self, var : VariableProtocol, cut : CutProtocol) -> Tuple[Any, Any, Any, np.dtype]:
        # override range to use streaming method

        if hasattr(var, '_wrt'):
            target = var._wrt # type: ignore
        else:
            target = var

        varexpr = target.to_pyarrow_expression()
        if varexpr is None:
            raise RuntimeError("Variable %s does not have a valid pyarrow expression"%(target.key))

        return self.streaming_get_range(
            varexpr,
            cut.to_pyarrow_expression()
        ) # type: ignore        
    
    @override 
    def get_unique(self, var : VariableProtocol, cut : CutProtocol) -> np.ndarray:
        varexpr = var.to_pyarrow_expression()
        if varexpr is None:
            raise RuntimeError("Variable %s does not have a valid pyarrow expression"%(var.key))

        unique_values = self.streaming_get_unique(
            varexpr,
            cut.to_pyarrow_expression()
        )

        return np.array(list(unique_values))

    @override
    def fill_hist(self,
                  variable: VariableProtocol, 
                  cut: CutProtocol, 
                  weight : VariableProtocol,
                  axis : Any) -> Any:
       
        if isinstance(variable, RateVariable):
            Hpass = hist.Hist(
                axis,
                storage=hist.storage.Weight()
            )
            Hfail = hist.Hist(
                axis,
                storage=hist.storage.Weight()
            )

            wrt = variable._wrt.to_pyarrow_expression()
            if wrt is None:
                raise RuntimeError("Variable %s does not have a valid pyarrow expression"%(variable._wrt.key))

            binaryvar = variable._binaryfield.to_pyarrow_expression()
            if binaryvar is None:
                raise RuntimeError("Variable %s does not have a valid pyarrow expression"%(variable._binaryfield.key))

            passcut = cut.to_pyarrow_expression()
            if passcut is None:
                passcut = variable._binaryfield.to_pyarrow_expression()
            else:
                passcut = pc.and_kleene(passcut, binaryvar)

            wtvar = weight.to_pyarrow_expression()
            if wtvar is None:
                raise RuntimeError("Variable %s does not have a valid pyarrow expression"%(weight.key))

            self.streaming_fill_histogram(
                Hpass,
                {axis.name : wrt},
                wtvar,
                passcut
            )

            failcut = cut.to_pyarrow_expression()
            if failcut is None:
                failcut = variable._binaryfield.to_pyarrow_expression()
            else:
                failcut = pc.and_not_kleene(failcut, binaryvar)

            self.streaming_fill_histogram(
                Hfail,
                {axis.name : wrt},
                wtvar,
                failcut
            )

            self._H = RateHistStruct(Hpass, Hfail)

        elif isinstance(variable, ProfileVariable):
            raise NotImplementedError("ProfileStruct is not implemented")
            #self._H = ProfileHistStruct(
            #    val,
            #    [axis]
            #)
        else:
            self._H = hist.Hist(
                axis,
                storage=hist.storage.Weight()
            )

            valvar = variable.to_pyarrow_expression()
            if valvar is None:
                raise RuntimeError("Variable %s does not have a valid pyarrow expression"%(variable.key))
            
            wtvar = weight.to_pyarrow_expression()
            if wtvar is None:
                raise RuntimeError("Variable %s does not have a valid pyarrow expression"%(weight.key))
            
            cutvar = cut.to_pyarrow_expression()

            print("filling hist")
            print("\tstarting", self._H.sum())
            print("\twtvar", wtvar)
            print("\tcutvar", cutvar)
            print("\tvalvar", valvar)
            print()

            self.streaming_fill_histogram(
                self._H,
                {axis.name : valvar},
                wtvar,
                cutvar
            )
        
        return self._H

    def streaming_get_range(self, variable : ds.Expression,
                            mask : ds.Expression | None,
                            batch_size : int = 1 << 20,
                            batch_readahead : int = 1,
                            fragment_readahead : int = 1,
                            use_threads : bool = True):
        # Specialized method to get the range of a variable in a streaming way, without loading the entire dataset into memory.
        # This is useful for very large datasets that cannot fit into memory.

        # Create an iterator over the dataset in batches
        columns = {'var' : variable}

        iterator = self._dataset.to_batches(
            columns = columns,
            filter = mask,
            batch_size = batch_size,
            batch_readahead = batch_readahead,
            fragment_readahead = fragment_readahead,
            use_threads = use_threads,
        )
        from tqdm import tqdm

        iterator = tqdm(iterator, desc='%s: range'%(self._key), unit='batch')
        minval = np.nan
        minval2 = np.nan
        maxval = np.nan
        dtype = np.float32
        for batch in iterator:

            var_array = np.asarray(batch['var'])
            if np.sum(np.isfinite(var_array)) == 0:
                batch_min = np.nan
                batch_min2 = np.nan
                batch_max = np.nan
            else:
                batch_min = np.nanmin(var_array)
                batch_max = np.nanmax(var_array)
                if np.sum(var_array > 0) > 0:
                    batch_min2 = np.nanmin(var_array[var_array > 0])
                else:
                    batch_min2 = np.nan

                if np.isnan(minval) or batch_min < minval:
                    minval = batch_min
                if np.isnan(maxval) or batch_max > maxval:
                    maxval = batch_max
                if np.isnan(minval2) or batch_min2 < minval2:
                    minval2 = batch_min2
                
                dtype = var_array.dtype

        return minval, minval2, maxval, dtype

    def streaming_get_unique(self, variable : ds.Expression,
                             mask : ds.Expression | None,
                             batch_size : int = 1 << 20,
                             batch_readahead : int = 1,
                             fragment_readahead : int = 1,
                             use_threads : bool = True):

        # Create an iterator over the dataset in batches
        columns = {'var' : variable}
        iterator = self._dataset.to_batches(
            columns = columns,
            filter = mask,
            batch_size = batch_size,
            batch_readahead = batch_readahead,
            fragment_readahead = fragment_readahead,
            use_threads = use_threads,
        )
        from tqdm import tqdm

        unique_values = set()
        iterator = tqdm(iterator, desc='%s: unique vals'%(self._key), unit='batch')
        for batch in iterator:
            var_array = np.asarray(batch['var'])
            unique_values.update(np.unique(var_array))

        return unique_values


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
        else:
            columns['weight'] = pc.scalar(self._weight)
            
        iterator = self._dataset.to_batches(
            columns = columns,
            filter = mask,
            batch_size = batch_size,
            batch_readahead = batch_readahead,
            fragment_readahead = fragment_readahead,
            use_threads = use_threads,
        )
        from tqdm import tqdm
        iterator = tqdm(iterator, desc='%s: Filling histogram'%self._key, unit='batch')
        for batch in iterator:
            # Evaluate the variables and weights for the current batch
            H.fill(**{name : value for name, value in zip(batch.column_names, batch.columns)})