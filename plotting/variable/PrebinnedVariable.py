from turtle import up
from plotting.typing.Protocols import VariableProtocol
from simon_mpl_util.plotting.typing.Protocols import PrebinnedOperationProtocol
from util.sanitization import maybe_valcov_to_definitely_valcov
from .VariableBase import VariableBase
from typing import Sequence, assert_never, override, List
import numpy as np

class BasicPrebinnedVariable(VariableBase):
    def __init__(self):
        pass #stateless
    
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
        return cut.evaluate(dataset)

    @property 
    def key(self):
        return "PREBINNED"

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")

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
        
        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, thelen, thedtype = maybe_valcov_to_definitely_valcov(evaluated)

        #after type checks we can be confident that
        #hist = np.ndarray with shape (thelen,), or else None
        #cov = np.ndarray with shape (thelen, thelen), or else None

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

        jacobian = np.ones(shape = (thelen,), dtype= thedtype)
        for key in binning.axis_names:
            jacobian *= widths[key].ravel()

        jacobian[jacobian == 0] = 1.0 #avoid division by zero

        if hist is not None:
            density_hist = hist / jacobian

        if cov is not None:
            cov_jacobian = np.outer(jacobian, jacobian)
            density_cov = cov / cov_jacobian

        if hist is None and cov is None:
            raise ValueError("This shouldn't be possible. Somehow both val and cov are None?")
        elif hist is None:
            return density_cov # pyright: ignore[reportPossiblyUnboundVariable]
        elif cov is None:
            return density_hist # pyright: ignore[reportPossiblyUnboundVariable]
        else:
            return density_hist, density_cov # pyright: ignore[reportPossiblyUnboundVariable]

    def __eq__(self, other) -> bool:
        if not isinstance(other, WithJacobian):
            return False
        return (self._var == other._var and
                self._radial_coords == other._radial_coords and
                self._clip_negativeinf == other._clip_negativeinf and
                self._clip_positiveinf == other._clip_positiveinf)

    @property 
    def key(self):
        return "WithJacobian(%s)" % self._var.key

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
        
        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, _, _ = maybe_valcov_to_definitely_valcov(evaluated)
        if hist is None:
            raise ValueError("Can't NormalizerPerBlock without histogram values!")
        
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
    def key(self):
        return "NormalizePerBlock(%s)" % self._var.key

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")

class CorrelationFromCovariance(VariableBase):
    def __init__(self, variable : VariableProtocol):
        self._var = variable

    @property
    def _natural_centerline(self):
        return 0

    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CorrelationFromCovariance):
            return False
        return (self._var == other._var)
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("PrebinnedDensityVariable requires a PrebinnedOperationProtocol cut")

        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, _, _ = maybe_valcov_to_definitely_valcov(evaluated)

        if cov is None:
            raise RuntimeError("CorrelationFromCovariance needs covariance!!")
        
        errs = np.sqrt(np.diag(cov))
        outer = np.outer(errs, errs)
        outer[outer==0] = 1 #avoid divide by zero

        correl = cov/outer

        if hist is not None:
            vals = hist/errs 
            return vals, correl
        else:
            return correl

    @property 
    def key(self):
        return "CorrelationFromCovariance(%s)" % self._var.key

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")
    
class _ExtractCovarianceMatrix(VariableBase):
    def __init__(self, variable : VariableProtocol):
        self._var = variable

    @property
    def _natural_centerline(self):
        return 0

    @property
    def columns(self):
        return []
    
    @property
    def prebinned(self) -> bool:
        return True
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, _ExtractCovarianceMatrix):
            return False
        return (self._var == other._var)
    
    @property 
    def key(self):
        return "Cov(%s)" % self._var.key

    def set_collection_name(self, collection_name):
        raise ValueError("Prebinned Variables do not support set_collection_name")
    
    def evaluate(self, dataset, cut):
        if not isinstance(cut, PrebinnedOperationProtocol):
            raise ValueError("ExtractCovarianceMatrix requires a PrebinnedOperationProtocol cut")

        evaluated = self._var.evaluate(dataset, cut)
        hist, cov, _, _ = maybe_valcov_to_definitely_valcov(evaluated)

        if cov is None:
            raise RuntimeError("ExtractCovarianceMatrix needs covariance!!")
        
        return cov