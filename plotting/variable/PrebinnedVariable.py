from turtle import up
from simon_mpl_util.plotting.typing.Protocols import PrebinnedOperationProtocol
from .VariableBase import VariableBase
from typing import Sequence, override
import numpy as np

class BasicPrebinnedVariable(VariableBase):
    def __init__(self):
        pass #stateless
    
    @property
    @override
    def _natural_centerline(self):
        return None
    
    @property 
    @override
    def columns(self):
        return []
    
    @property
    @override
    def prebinned(self) -> bool:
        return True
    
    @override
    def evaluate(self, dataset, cut):
        return cut.evaluate(dataset)

    @property 
    @override
    def key(self):
        return "PREBINNED"

    @override    
    def set_collection_name(self, collection_name):
        raise ValueError("PrebinnedVariable does not support set_collection_name")

    @override
    def __eq__(self, other) -> bool:
        return isinstance(other, BasicPrebinnedVariable)
    
class PrebinnedDensityVariable(VariableBase):
    def __init__(self, radial_coords : Sequence[str]):
        self._radial_coords = radial_coords

    @property
    def _natural_centerline(self):
        return None
    
    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("PrebinnedDensityVariable requires a PrebinnedOperationProtocol cut")
        hist = cut.evaluate(dataset)
        binning = cut.resulting_binning(dataset)

        lower_edges = binning.lower_edges()
        upper_edges = binning.upper_edges()

        widths = {}
        for key in binning.axis_names:
            if key in self._radial_coords:
                widths[key] = np.square(upper_edges[key]) - np.square(lower_edges[key])
            else:
                widths[key] = upper_edges[key] - lower_edges[key]

        jacobian = np.ones_like(hist)
        for key in binning.axis_names:
            jacobian *= widths[key]

        jacobian[jacobian == 0] = 1.0 #avoid division by zero
        density_hist = hist / jacobian
        return density_hist