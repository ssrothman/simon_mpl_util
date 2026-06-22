


import matplotlib.pyplot as plt
import numpy as np
from simonplot.config.lookuputil import check_auto_logx
from simonplot.plottables.Datasets import DatasetComparison
from simonplot.typing.Protocols import AutoBinningProtocol, BaseBinningProtocol, BaseDatasetProtocol, BasicBinningProtocol, CutProtocol, DefaultBinningProtocol, VariableProtocol
from typing import Tuple
import os.path
from matplotlib.colors import LogNorm, Normalize

from simonplot.util.common import add_cms_legend, add_text, make_oneax, setup_canvas, savefig
from simonplot.variable.Variable import ProfileVariable, RateVariable

def draw_hist2D(variableX : VariableProtocol, variableY : VariableProtocol,
                cut : CutProtocol,
                weight : VariableProtocol,
                dataset : BaseDatasetProtocol,
                binningX : BaseBinningProtocol,
                binningY : BaseBinningProtocol,
                extratext : str | None,
                textloc : str | int | Tuple[float, float, str, str],
                logx : bool | None,
                logy : bool | None,
                logc : bool | None,
                no_lumi_normalization : bool,
                output_folder : str | None,
                output_prefix : str | None,
                override_filename : str | None,
                override_cbarlabel : str | None):
    if logx is None and not variableX.prebinned:
        if isinstance(variableX, (RateVariable, ProfileVariable)):
            logx = check_auto_logx(variableX.xkey)
        else:
            logx = check_auto_logx(variableX.key)

    if logy is None and not variableY.prebinned:
        if isinstance(variableY, (RateVariable, ProfileVariable)):
            logy = check_auto_logx(variableY.xkey)
        else:
            logy = check_auto_logx(variableY.key)

    if isinstance(binningX, AutoBinningProtocol):
        if logx:
            transform = 'log'
        else:
            transform = None
        
        axisX = binningX.build_auto_axis([variableX], [cut], [dataset], transform=transform) 
    elif isinstance(binningX, DefaultBinningProtocol):
        axisX = binningX.build_default_axis(variableX)
    elif isinstance(binningX, BasicBinningProtocol):
        axisX = binningX.build_axis(variableX)
    else:
        raise ValueError("Unsupported binning type for X axis")
    
    if isinstance(binningY, AutoBinningProtocol):
        if logy:
            transform = 'log'
        else:
            transform = None

        axisY = binningY.build_auto_axis([variableY], [cut], [dataset], transform=transform) # type: ignore
    elif isinstance(binningY, DefaultBinningProtocol):
        axisY = binningY.build_default_axis(variableY)
    elif isinstance(binningY, BasicBinningProtocol):
        axisY = binningY.build_axis(variableY)
    else:
        raise ValueError("Unsupported binning type for Y axis")
    
    fig = setup_canvas()
    ax_main = make_oneax(fig)

    if dataset.isMC:
        add_cms_legend(ax_main, False)
    else:
        add_cms_legend(ax_main, True, lumi=dataset.lumi)

    # actually draw the plot
    H = dataset.fill_hist_2D(
        variableX, variableY,
        cut, weight,
        axisX, axisY
    )
    mat = H.values(flow=False)
    xedges = H.axes[0].edges
    yedges = H.axes[1].edges

    if xedges[0] == -np.inf:
        print("WARNING: clipping underflow bin on x axis")
        xedges = xedges[1:]
        mat = mat[1:, :]
    if yedges[0] == -np.inf:
        print("WARNING: clipping underflow bin on y axis")
        yedges = yedges[1:]
        mat = mat[:, 1:]

    if logc is None:
        if np.min(mat) * np.max(mat) < 0:
            logc = False
        else:
            ratio = np.abs(np.max(mat) / np.min(mat[mat != 0]))
            if ratio > 10:
                logc = True
            else:
                logc = False

    cmap = 'plasma'
    if logc:
        normobj = LogNorm()
    else:
        normobj = Normalize(vmin=0)

    artist = plt.pcolormesh(
        xedges, yedges,
        mat.T,
        cmap = cmap,
        norm = normobj,
        rasterized = True
    )

    cbar = fig.colorbar(artist, ax=ax_main)
    
    if override_cbarlabel is not None:
        cbar.set_label(override_cbarlabel)
    else:
        if isinstance(dataset, DatasetComparison):
            clabel = dataset.ylabel
        else:
            clabel = '$\\frac{dN}{d(%s)d(%s)}$' % (variableX.key, variableY.key)

        cbar.set_label(clabel)

    the_xlabel = variableX.label
    the_ylabel = variableY.label
    ax_main.set_xlabel(the_xlabel)
    ax_main.set_ylabel(the_ylabel)

    if logx:
        ax_main.set_xscale('log')
    if logy:
        ax_main.set_yscale('log')

    add_text(ax_main, cut, extratext, loc=textloc)
    
    fig.tight_layout()

    if output_folder is not None:
        if override_filename is not None:
            output_path = os.path.join(output_folder, override_filename)
        else:
            if output_prefix is None:
                output_path = os.path.join(output_folder, 'hist2D')
            else:
                output_path = os.path.join(output_folder, output_prefix)

            output_path += '_X-VAR-%s' % variableX.key
            output_path += '_Y-VAR-%s' % variableY.key
            output_path += '_CUT-%s' % cut.key
            output_path += '_WGT-%s' % weight.key
            output_path += '_DSET-%s' % dataset.key
            if logx:
                output_path += '_LOGX'
            if logy:
                output_path += '_LOGY'
            if logc:
                output_path += '_LOGC'

        savefig(fig, output_path)
    else:
        plt.show()
    
    plt.close(fig)
