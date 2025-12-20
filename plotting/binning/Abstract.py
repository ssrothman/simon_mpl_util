import hist
from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning

from simon_mpl_util.plotting.variable.Abstract import AbstractVariable
from simon_mpl_util.plotting.plottables.Abstract import AbstractDataset
from simon_mpl_util.plotting.cut.Abstract import AbstractCut, PrebinnedOperation

from typing import List, Union

class AbstractBinning:
    def build_axis(self, variable : AbstractVariable) -> hist.axis.AxesMixin:
        raise NotImplementedError()
    
    def build_default_axis(self, variable: AbstractVariable) -> hist.axis.AxesMixin:
        raise NotImplementedError()
    
    def build_auto_axis(self, 
                        variables: List[AbstractVariable], 
                        cuts: List[AbstractCut], 
                        datasets: List[AbstractDataset], 
                        transform: Union[str, None]=None) -> hist.axis.AxesMixin:
        raise NotImplementedError()

    def build_prebinned_axis(self, 
                             dataset : AbstractDataset,
                             cut : PrebinnedOperation) -> ArbitraryBinning:
        raise NotImplementedError()

