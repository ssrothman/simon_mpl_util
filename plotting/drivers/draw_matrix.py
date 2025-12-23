import os
from typing import Union, List, Literal, get_args

import matplotlib
from matplotlib.colors import Normalize, SymLogNorm, LogNorm

from simon_mpl_util.plotting.variable.VariableBase import AbstractVariable
from simon_mpl_util.plotting.cut.CutBase import PrebinnedOperation
from simon_mpl_util.plotting.plottables.DatasetBase import AbstractPrebinnedDataset
from simon_mpl_util.plotting.binning.BinningBase import AbstractBinning

from simon_mpl_util.plotting.util.common import add_text, label_from_binning, make_fancy_prebinned_labels, setup_canvas, make_oneax, savefig, add_cms_legend

import numpy as np
import matplotlib.pyplot as plt

_ALLOWED_NORMS = Literal["none", "ax1", "ax2", "correl"]

def draw_matrix(cut: PrebinnedOperation, 
                dataset: AbstractPrebinnedDataset,
                binning : AbstractBinning,
                extratext : Union[str, None] = None,
                norm: _ALLOWED_NORMS = "none",
                sym : Union[bool, None] = None,
                logc : bool = False,
                output_folder: Union[str, None] = None,
                output_prefix: Union[str, None] = None):
    
    #enforce norm options
    valid_norms = get_args(_ALLOWED_NORMS)  
    if norm not in valid_norms:
        raise ValueError("Invalid norm '%s', must be one of %s" % (norm, valid_norms))

    #enforce prebinned
    if not binning.kind == 'prebinned':
        raise TypeError("draw_matrix only supports prebinned binning")
   
    #correl normalization only makes sense for covariance matrices
    #and ax1, ax2 normalization only make sense for transfer matrices
    if dataset.quantitytype == 'covariance':
        if norm in ['ax1', 'ax2']:
            raise ValueError("ax1 and ax2 normalization not valid for covariance matrices")
    elif dataset.quantitytype == 'transfer':
        if norm == 'correl':
            raise ValueError("correl normalization not valid for transfer matrices")

    #get resulting binning
    axis = binning.build_prebinned_axis(dataset, cut)

    #get matrix to plot
    mat = cut.evaluate(dataset)
    mat = _maybe_normalize(mat, norm)

    #automatically determine if values are (conceptually) symmetric about zero
    if sym is None:
        if norm == 'correl':
            sym = True
        elif np.min(mat) < 0 and np.max(mat) > 0:
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
    if dataset.isMC:
        add_cms_legend(ax, False)
    else:
        add_cms_legend(ax, True, lumi=dataset.lumi)

    artist = ax.pcolormesh(mat, cmap=cmap, norm=normobj, rasterized=True)

    the_xlabel = label_from_binning(axis)
    ax.set_xlabel(the_xlabel)
    ax.set_ylabel(the_xlabel)

    cbar = fig.colorbar(artist, ax=ax)
    if dataset.quantitytype == 'transfer':
        if norm == 'none':
            cbarlabel = 'Transfer matrix'
        elif norm == 'ax1':
            cbarlabel = 'Row-normalized transfer matrix [TO DO: CHECK IF THIS IS GEN OR RECO]'
        elif norm == 'ax2':
            cbarlabel = 'Column-normalized transfer matrix [TO DO: CHECK IF THIS IS GEN OR RECO]'
        else:
            cbarlabel = 'Malformed quantity lol'
    elif dataset.quantitytype == 'covariance':
        if norm == 'none':
            cbarlabel = 'Covariance'
        elif norm == 'correl':
            cbarlabel = 'Correlation'
        else:
            cbarlabel = 'Malformed quantity lol'
    else:
        cbarlabel = 'Malformed quantity lol'

    cbar.set_label(cbarlabel)

    add_text(ax, cut, extratext)
    
    make_fancy_prebinned_labels(ax, None, axis, 'x')
    make_fancy_prebinned_labels(ax, None, axis, 'y')

    ax.set_box_aspect(1)

    fig.tight_layout()

    if output_folder is not None:
        if output_prefix is None:
            output_path = os.path.join(output_folder, 'matrix')
        else:
            output_path = os.path.join(output_folder, output_prefix)

        output_path += '_CUT-%s' % cut.key
        output_path += '_DSET-%s' % dataset.key
        output_path += '_NORM-%s' % norm
        
        if logc:
            output_path += '_LOGC'

        savefig(fig, output_path)
    else:
        plt.show()

    plt.close(fig)
    
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