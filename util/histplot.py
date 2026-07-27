import matplotlib.pyplot as plt
import hist
import numpy as np

from simonplot.util.comparison import ComparisonHistStruct
from simonplot.util.profile import ProfileHistStruct
from simonplot.util.rate import RateHistStruct
from simonpy.AbitraryBinning import ArbitraryBinning

def _call_grayboxes(ax, edges, plotvals, yerr, **kwargs):
    low = plotvals - yerr
    high = plotvals + yerr
    
    if 'color' in kwargs:
        del kwargs['color']
    if 'c' in kwargs:
        del kwargs['c']

    return ax.stairs(
        high, edges,
        baseline=low,
        fill=True,
        color='gray',
        alpha=0.5,
        **kwargs
    )

def _call_errorbar(ax, x, y, xerr, yerr, marker=None, **kwargs):
    if marker is not None:
        kwargs['markersize'] = 8
    else:
        kwargs['markersize'] = 4

    if marker is None:
        marker = 'o'

    return ax.errorbar(
        x, y, xerr = xerr, yerr = yerr,
        fmt = marker, capsize=1, 
        **kwargs
    )

def _call_stairs(ax, edges, plotvals, fillbetween, **kwargs):
    return ax.stairs(
        plotvals, edges, 
        baseline = fillbetween, fill=True, 
        **kwargs
    )

def simon_histplot_rate(H, ax=None, **kwargs):
    if len(H.axes) != 2:
        raise ValueError("histplot_rate only supports 2D histograms")

    if ax is None:
        ax = plt.gca()

    vals = H.values().copy()
    errs = np.sqrt(H.variances()).copy()

    passvals = vals[1, :]
    passerrs = errs[1, :]

    failvals = vals[0, :]
    failerrs = errs[0, :]

    total = passvals + failvals

    rate = passvals / total
    rateerr = np.sqrt(
        np.square((1-rate)*passerrs/total) +
        np.square(rate*failerrs/total)
    )

    edges = H.axes[1].edges
    centers = H.axes[1].centers
    widths = np.abs(H.axes[1].widths)

    if type(H.axes[1]) is hist.axis.Integer:
        centers -= 0.5
        edges -= 0.5

    return _call_errorbar(ax, centers, rate, widths/2, rateerr, **kwargs)

def _simon_histplot(vals, errs, edges, centers, widths,
                    ax=None, density=False, fillbetween = None, 
                    dont_divide_by_width = False, 
                    marker = None, gray_boxes = False, **kwargs):
    if density:
        N = np.sum(vals)
        vals /= N
        errs /= N

    plotvals = vals/widths if not dont_divide_by_width else vals
    ploterrs = errs/widths if not dont_divide_by_width else errs

    if fillbetween is not None:
        # clip flow bins
        # NB fillbetween does NOT need to be clipped
        # as it is built from the (already clipped) return of THIS FUNCTION
        if edges[0] == -np.inf:
            edges = edges[1:]
            plotvals = plotvals[1:]

        if edges[-1] == +np.inf:
            edges = edges[:-1]
            plotvals = plotvals[:-1]

        if marker is not None:
            raise ValueError("Cannot use marker with fillbetween")
        
        artist = _call_stairs(ax, edges, plotvals+fillbetween, fillbetween, **kwargs)
        return artist, plotvals+fillbetween
    elif gray_boxes:
        if marker is not None:
            raise ValueError("Cannot use marker with gray_boxes")
        
        artist = _call_grayboxes(ax, edges, plotvals, ploterrs, **kwargs)
        return artist, plotvals
    else:
        artist = _call_errorbar(ax, centers, plotvals, widths/2, ploterrs, marker=marker, **kwargs)
        return artist, plotvals

def simon_histplot_arbitrary(vals : np.ndarray, cov : np.ndarray,
                             binning : ArbitraryBinning, 
                             ax=None, density=False, fillbetween = None, 
                             jitter_i : int | None = None, 
                             jitter_N : int | None = None, 
                             jitter_log : bool | None = None,
                             gray_boxes : bool = False,
                             marker : str | None = None,
                             **kwargs):
    
    vals = vals.copy()
    errs = np.sqrt(np.diag(cov))

    if binning.Nax == 1:
        axname = binning.axis_names[0]
        edges = binning.edges[axname]

    else:
        edges = np.arange(len(vals)+1)-0.5

    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.abs(edges[1:] - edges[:-1])

    if jitter_i is not None and jitter_N is not None:
        # jitter x coordinates to avoid overlapping points
        # jitter is done +-1/10 of the bin width

        jitterint = jitter_i / (jitter_N-1) * 2 - 1

        if jitter_log is None or not jitter_log:
            # linear scale, so we want to jitter in linear space
            jitter_width = widths / 10
            jitterfloat = jitterint * jitter_width
            centers += jitterfloat

        else:
            # we are in log scale, so we want to jitter in log space
            jitter_width =  np.log10(edges[1:]) - np.log10(edges[:-1])
            jitter_width /= 10
            jitterfloat = jitterint * jitter_width
            if edges[0] == 0:
                jitterfloat[0] = jitterfloat[1]  # avoid log10(0) -> -inf
                widths[0] *= np.power(10, jitterfloat[0])
            centers *= np.power(10, jitterfloat)

    return _simon_histplot(vals, errs, edges, centers, widths,
                           ax=ax, density=density, fillbetween=fillbetween, 
                           marker=marker, gray_boxes=gray_boxes, **kwargs)

def simon_histplot(H, ax=None, density=False, fillbetween = None, 
                   jitter_i=None, jitter_N=None, jitter_log=None,
                   marker=None, gray_boxes=False, **kwargs):
    if len(H.axes) != 1:
        raise ValueError("histplot only supports 1D histograms")

    if ax is None:
        ax = plt.gca()

    vals = H.values().copy()
    errs = np.sqrt(H.variances()).copy()

    edges = H.axes[0].edges
    centers = H.axes[0].centers
    widths = np.abs(H.axes[0].widths)

    if jitter_i is not None and jitter_N is not None:
        # jitter x coordinates to avoid overlapping points
        # jitter is done +-1/10 of the bin width

        jitterint = jitter_i / (jitter_N-1) * 2 - 1

        if jitter_log is None or not jitter_log:
            # linear scale, so we want to jitter in linear space
            jitter_width = widths / 10
            jitterfloat = jitterint * jitter_width
            centers += jitterfloat

        else:
            # we are in log scale, so we want to jitter in log space
            jitter_width =  np.log10(edges[1:]) - np.log10(edges[:-1])
            jitter_width /= 10
            jitterfloat = jitterint * jitter_width
            if edges[0] == 0:
                jitterfloat[0] = jitterfloat[1]  # avoid log10(0) -> -inf
                widths[0] *= np.power(10, jitterfloat[0])
            centers *= np.power(10, jitterfloat)


    if type(H.axes[0]) is hist.axis.Integer:
        centers -= 0.5
        edges -= 0.5

    if isinstance(H, ComparisonHistStruct) and H.mode == 'ratio':
        kwargs['dont_divide_by_width'] = True
    elif isinstance(H, (RateHistStruct, ProfileHistStruct)):
        kwargs['dont_divide_by_width'] = True
    
    
    if 'dont_divide_by_width' in kwargs:
        dont_divide_by_width = kwargs.pop('dont_divide_by_width')
    else:
        dont_divide_by_width = False
        
    return _simon_histplot(vals, errs, edges, centers, widths,
                           ax=ax, density=density, fillbetween=fillbetween,
                           dont_divide_by_width = dont_divide_by_width,
                           marker=marker, gray_boxes=gray_boxes,
                           **kwargs)

def _simon_histplot_ratio(vals_num, errs_num,
                          vals_denom, errs_denom,
                          edges, centers, widths,
                          ax=None, 
                          density=False, pulls=False, **kwargs):
    
    if density:
        Nnum = np.sum(vals_num)
        Ndenom = np.sum(vals_denom)

        errs_num /= Nnum
        vals_num /= Nnum

        errs_denom /= Ndenom
        vals_denom /= Ndenom

    vals_num /= widths
    errs_num /= widths

    vals_denom /= widths
    errs_denom /= widths

    with np.errstate(divide='ignore', invalid='ignore'): #ignore warnings from 0/0 operations. These return NaN, which are handled correctly downstream
        ratio = vals_num / vals_denom

        ratio_err = np.sqrt(
                np.square(errs_num/vals_denom) + np.square(vals_num*errs_denom/np.square(vals_denom))
        ) 

    if pulls:
        ratio = ratio-1
        ratio = ratio/ratio_err
        ratio_err = np.ones_like(ratio)
    
    return _call_errorbar(ax, centers, ratio, widths/2, ratio_err, **kwargs), ratio, ratio_err

def simon_histplot_ratio_arbitrary(num, denom,
                                   binning : ArbitraryBinning,
                                   ax=None,
                                   density=False, pulls=False, **kwargs):    
    vals_num, cov_num = num
    vals_denom, cov_denom = denom

    vals_num = vals_num.copy()
    vals_denom = vals_denom.copy()
    
    errs_num = np.sqrt(np.diag(cov_num))
    errs_denom = np.sqrt(np.diag(cov_denom))

    if binning.Nax == 1:
        axname = binning.axis_names[0]
        edges = binning.edges[axname]
    else:
        edges = np.arange(len(vals_num)+1)-0.5

    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.abs(edges[1:] - edges[:-1])

    return _simon_histplot_ratio(vals_num, errs_num,
                                 vals_denom, errs_denom,
                                 edges, centers, widths,
                                 ax=ax, density=density, pulls=pulls, **kwargs)

def simon_histplot_ratio(Hnum, Hdenom, ax=None, 
                         density=False, pulls=False, **kwargs):
    
    if len(Hnum.axes) != 1 or len(Hdenom.axes) != 1:
        raise ValueError("histplot only supports 1D histograms")

    if Hnum.axes[0] != Hdenom.axes[0]:
        raise ValueError("histograms must have the same axes")

    if ax is None:
        ax = plt.gca()

    vals_num = Hnum.values().copy()
    errs_num = np.sqrt(Hnum.variances()).copy()

    vals_denom = Hdenom.values().copy()
    errs_denom = np.sqrt(Hdenom.variances()).copy()

    edges = Hnum.axes[0].edges 
    centers = Hnum.axes[0].centers
    widths = np.abs(Hnum.axes[0].widths)

    if type(Hnum.axes[0]) is hist.axis.Integer:
        centers -= 0.5

    return _simon_histplot_ratio(vals_num, errs_num,
                                 vals_denom, errs_denom,
                                 edges, centers, widths,
                                 ax=ax, density=density, pulls=pulls, **kwargs)