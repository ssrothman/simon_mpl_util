from .Cut import AbstractCut
from typing import List
from ..datasets import AbstractDataset, PrebinnedDataset
from ..AribtraryBinning import ArbitraryBinning
from ..SetupConfig import lookup_axis_label
import numpy as np

def xlabel_from_binning(binning : ArbitraryBinning) -> str:
    if binning.Nax == 1:
        return lookup_axis_label(binning.axis_names[0])
    else:
        return '@'.join([lookup_axis_label(ax) for ax in binning.axis_names]) + " bin index"

class PrebinnedOperation(AbstractCut):
    @property
    def columns(self):
        return []    
    
    def resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        if hasattr(self, '_resulting_binning'):
            return self._resulting_binning # pyright: ignore[reportAttributeAccessIssue]
        else:
            return self._compute_resulting_binning(binning)
        
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        raise NotImplementedError("PrebinnedOperation subclasses must implement compute_resulting_binning method")
    
class NoopOperation(PrebinnedOperation):
    @property
    def key(self):
        return "NOOP"
    
    def _auto_plottext(self):
        return ''

    def __eq__(self, other):
        return isinstance(other, NoopOperation)

    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("NoopOperation can only be applied to PrebinnedDataset")
        
        return dataset.values, dataset.cov

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        self._resulting_binning = binning
        return binning

class ProjectionOperation(PrebinnedOperation):
    def __init__(self, axes : List[str]):
        self._axes = axes

    @property
    def key(self):
        return "PROJECT(%s)" % "-".join(str(ax) for ax in self._axes)

    def _auto_plottext(self):
        return 'Integrated over %s' % ", ".join(self._axes)

    def __eq__(self, other):
        if not isinstance(other, ProjectionOperation):
            return False
        return self._axes == other._axes

    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("ProjectionOperation can only be applied to PrebinnedDataset")
        
        return dataset.project(self._axes)[:2]

    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        result = binning
        empty_data = np.empty(binning.total_size)
        for ax in self._axes:
            empty_data, result = result.project_out(empty_data, ax)
        self._resulting_binning = result
        return result

class SliceOperation(PrebinnedOperation):
    def __init__(self, edges):
        self._edges = edges

    @property
    def key(self):
        slicestr = ''
        for name, edges in self._edges.items():
            slicestr+='%s-%sto%s_' % (name, edges[0], edges[1])
        if slicestr[-1] == '_':
            slicestr = slicestr[:-1]
        return "SLICE(%s)" % slicestr
    
    def _auto_plottext(self):
        texts = []
        for name, edges in self._edges.items():
            texts.append('%s in [%s, %s]' % (name, edges[0], edges[1]))
        return ' and '.join(texts)

    def __eq__(self, other):
        if not isinstance(other, SliceOperation):
            return False
        return self.key == other.key
    
    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("SliceOperation can only be applied to PrebinnedDataset")
        
        return dataset.slice(self._edges)
    
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        raise NotImplementedError("TO DO")

class ProjectAndSliceOperation(PrebinnedOperation):
    def __init__(self, axes : List[str], edges):
        self._projection = ProjectionOperation(axes)
        self._slice = SliceOperation(edges)

    @property
    def key(self):
        return "%s_%s" % (self._projection.key, self._slice.key)

    def __eq__(self, other):
        if not isinstance(other, ProjectAndSliceOperation):
            return False
        return self._projection == other._projection and self._slice == other._slice

    def _auto_plottext(self):
        raise NotImplementedError("TO DO")

    def evaluate(self, dataset):
        if not isinstance(dataset, PrebinnedDataset):
            raise TypeError("ProjectAndSliceOperation can only be applied to PrebinnedDataset")
        
        projected = self._projection.evaluate(dataset)
        return self._slice.evaluate(projected)
    
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        raise NotImplementedError("TO DO")
