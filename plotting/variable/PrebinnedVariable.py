from turtle import up
from plotting.typing.Protocols import VariableProtocol
from simon_mpl_util.plotting.typing.Protocols import PrebinnedOperationProtocol
from .VariableBase import VariableBase
from typing import Sequence, override, List
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
        raise ValueError("Prebinned Variables do not support set_collection_name")

    @override
    def __eq__(self, other) -> bool:
        return isinstance(other, BasicPrebinnedVariable)
    
class WithJacobian(VariableBase):
    def __init__(self, variable : VariableProtocol, 
                 radial_coords : Sequence[str],
                 clip_negativeinf : dict[str, float] = {},
                 clip_positiveinf : dict[str, float] = {}):
        self._var = variable
        self._radial_coords = radial_coords
        self._clip_negativeinf = clip_negativeinf
        self._clip_positiveinf = clip_positiveinf

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
        
        hist = self._var.evaluate(dataset, cut)
        binning = cut.resulting_binning(dataset)

        lower_edges = binning.lower_edges()
        upper_edges = binning.upper_edges()

        for key in self._clip_negativeinf:
            if key in lower_edges:
                edges = lower_edges[key]
                lower_edges[key] = np.where(edges == -np.inf, self._clip_negativeinf[key], edges)
                

        for key in self._clip_positiveinf:
            if key in upper_edges:
                edges = upper_edges[key]
                upper_edges[key] = np.where(edges == np.inf, self._clip_positiveinf[key], edges)

        widths = {}
        for key in binning.axis_names:
            if key in self._radial_coords:
                widths[key] = np.square(upper_edges[key]) - np.square(lower_edges[key])
            else:
                widths[key] = upper_edges[key] - lower_edges[key]

        if type(hist) is tuple:
            hist, cov = hist
        else:
            cov = None

        jacobian = np.ones_like(hist)
        for key in binning.axis_names:
            jacobian *= widths[key].ravel()

        jacobian[jacobian == 0] = 1.0 #avoid division by zero
        density_hist = hist / jacobian

        if cov is not None:
            density_jacobian = np.outer(jacobian, jacobian)
            density_cov = cov / density_jacobian
            return density_hist, density_cov
        else:
            return density_hist
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, WithJacobian):
            return False
        return (self._var == other._var and
                self._radial_coords == other._radial_coords and
                self._clip_negativeinf == other._clip_negativeinf and
                self._clip_positiveinf == other._clip_positiveinf)

    @property 
    @override
    def key(self):
        return "WithJacobian(%s)" % self._var.key

    @override    
    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")


class NormalizePerBlock(VariableBase):
    def __init__(self, variable : VariableProtocol, axes : List[str]):
        self._var = variable
        self._axes = axes

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
        
        hist = self._var.evaluate(dataset, cut)
        if type(hist) is tuple:
            hist, cov = hist
        else:
            cov = None

        binning = cut.resulting_binning(dataset)

        fluxes, shapes, _ = binning.get_fluxes_shapes(hist, self._axes)

        if cov is not None:
            _, covshapes, _ = binning.get_fluxes_shapes_cov2d(fluxes, shapes, cov,  self._axes)
            return shapes, covshapes
        else:
            return shapes
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, NormalizePerBlock):
            return False
        return (self._var == other._var and
                self._axes == other._axes)
    
    @property 
    @override
    def key(self):
        return "NormalizePerBlock(%s)" % self._var.key

    @override    
    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")

