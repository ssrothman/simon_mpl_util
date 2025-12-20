from typing import Union, List, Literal, get_args

from matplotlib.colors import Normalize, SymLogNorm, LogNorm

from simon_mpl_util.plotting.variable.Abstract import AbstractVariable
from simon_mpl_util.plotting.cut.Abstract import PrebinnedOperation
from simon_mpl_util.plotting.plottables.Abstract import AbstractPrebinnedDataset
from simon_mpl_util.plotting.binning.Abstract import AbstractBinning

from simon_mpl_util.plotting.util.common import setup_canvas, make_oneax, savefig

import numpy as np

_ALLOWED_NORMS = Literal["none", "ax1", "ax2", "correl"]

def draw_matrix(cut: PrebinnedOperation, 
                dataset: AbstractPrebinnedDataset,
                binning : AbstractBinning,
                extratext : Union[str, None] = None,
                norm: _ALLOWED_NORMS = "none",
                sym : Union[bool, None] = None,
                logc : bool = False,
                no_ratiopad : bool = False,
                output_folder: Union[str, None] = None,
                output_prefix: Union[str, None] = None):
    
    valid_norms = get_args(_ALLOWED_NORMS)  
    if norm not in valid_norms:
        raise ValueError("Invalid norm '%s', must be one of %s" % (norm, valid_norms))

    if not binning.kind == 'prebinned':
        raise TypeError("draw_matrix only supports prebinned binning")

    mat = cut.evaluate(dataset)
    mat = _maybe_normalize(mat, norm)

    if sym is None:
        if np.min(mat) < 0 and np.max(mat) > 0:
            sym = True
        else:
            sym = False

    if sym:
        cmap = 'coolwarm'
        extreme = np.max(np.abs(mat))
        if logc:
            normobj = SymLogNorm(
                vmin = -extreme,
                vmax = extreme,
                linthresh=extreme/1e3,
                linscale=1e-1,
            )
        else:
            normobj = Normalize(
                vmin = -extreme,
                vmax = extreme,
            )
    else:
        cmap = 'viridis'
        if logc:
            normobj = LogNorm()
        else:
            normobj = Normalize()
        
    fig = setup_canvas()
    ax = make_oneax(fig)

    ax.pcolormesh(mat, cmap=cmap, norm=normobj, rasterized=True)
    ax.set_aspect('equal')

    savefig(fig, 'test')
    
def _maybe_normalize(mat : np.ndarray,
                     norm : _ALLOWED_NORMS) -> np.ndarray:
    if norm == "none":
        return mat
    elif norm == "ax1":
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return mat / row_sums
    elif norm == "ax2":
        col_sums = mat.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        return mat / col_sums
    elif norm == "correl":
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Correlation normalization requires a square matrix")
        
        diag = np.sqrt(np.diag(mat))
        total_sum = np.outer(diag, diag)
        total_sum[total_sum == 0] = 1
        return mat / total_sum
    else:
        raise ValueError("Invalid norm '%s'" % norm)