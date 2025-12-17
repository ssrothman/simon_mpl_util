from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds

import numpy as np

class AbstractDataset:
    def ensure_columns(self, columns):
        raise NotImplementedError()

    def get_column(self, column_name, collection_name=None):
        raise NotImplementedError()

    def get_aknum_column(self, column_name):
        raise NotImplementedError()
    
    @property
    def num_rows(self):
        raise NotImplementedError()

class NanoEventsDataset(AbstractDataset):
    def __init__(self, fname, **options):
        #suppress warnings
        NanoAODSchema.warn_missing_crossrefs = False

        import coffea
        version = coffea._version.version_tuple
        if version[0] >= 2025 and version[1] >= 11 and version[2] >= 0:
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
    
    def get_aknum_column(self, column_name):
        return ak.to_numpy(ak.num(self._events[column_name]))
    
    @property
    def num_rows(self):
        return len(self._events)
    
class ParquetDataset(AbstractDataset):
    def __init__(self, path):
        self._dataset = ds.dataset(path, format="parquet")
        
    def ensure_columns(self, columns):
        self._table = self._dataset.to_table(columns=columns)
    
    def get_column(self, column_name, collection_name=None):
        if collection_name is not None:
            raise NotImplementedError("ParquetDataset does not support collection_name argument")
        
        return self._table[column_name].to_numpy()
    
    def get_aknum_column(self, column_name):
        raise NotImplementedError("ParquetDataset does not support ak.num columns")

    @property
    def num_rows(self):
        return self._table.num_rows