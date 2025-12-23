# Configuration of the plotting package

Many key aspects of the plot style are controlled by a configuration `json`, including axis labels, ratiopad style, and automatic setting of various preferences (such as logarithmic axes). 

## Default values and user configuration

Default values are stored in `simon_mpl_util/plotting/config/default.json`. For user configuration, the package looks for a user configuration in `${PWD}/simon_mpl_config.json` wherever the package is imported from, and overwrites default values with user-supplied configuration values. Any entries in the user configuration which do not match the structure of the defaults json will result in an error (for example mismatched datatypes or misspelled keys).

## Config structure

### Basic figure properties

Basic properties of the figure are controlled by

 - `figsize : [float, float]` - (x, y) dimensions of the default figure canvas, in inches
 - `cms_label : str` - the CMS approval label to use (e.g. "Work in Progress", "Preliminary", etc)
 
 ### Ratio pad style

 The ratio pad (under the main canvas) for `plot_histogram` calls is configured by the `ratiopad` dictionary. The options are


  - `ratiopad.height : float` - the ratio between the height of the ratio pad and the axes with the main plot
  - `ratiopad.hspace : float` - the padding between the main axes and the ratio pad
  
#### Automatic y-limits on the ratio pad

`plot_histogram` attempts to inteligently set the y-limits for the ratiopad. Simply including all of the data points sometimes makes the y range too large to include a large fluctuation in a bin with poor statistical support. Instead, I use the following approach:

1. Compute the ratio values and uncertainties in every bin.
2. Find the `ratiopad.auto_ylim.percentile` largest ratio uncertainty.
3. If this threshold is less than `ratiopad.auto_ylim.min_threshold`, clip the threshold.
4. Compute the envelope of all ratio values with uncertainty less than this threshold.
5. Add additional padding equal to the threshold value used (to ensure that the error bars fit in the plot) plus a configurable additional padding (`ratiopad.auto_ylim.padding`)

The free parameters are:
 
 - `ratiopad.auto_ylim.percentile : float` - the percentile ratio uncertainty to use as a threshold
 - `ratiopad.auto_ylim.min_threshold : float` - the minimum threshold below which all ratio points are guanteed to be on the plot
 - `ratiopad.auto_ylim.padding : float` - the (absolute) amount of extra padding to add on top of the default ylimits

Note that the axis limits are guarenteed to always be at least `[1-min_threshold-padding, 1+min_threshold+padding]`

### Fancy prebinned labels

For prebinned high-dimensional binnings, it is not always possible to label the plot axes (with any plotting driver) with physical quantities. If the binning after all projection, slice, etc. operations is one-dimensional, the axis range, ticks, etc will be the relevant physical quantity. If, however, the binning is at least 2D, then it is not obvious how to label the axes. "Fancy prebinned labels" refers to my attempt to label explicitely the blocks corresponding to the outermost axis in the binning. 

For example, say we have a 3D prebinned dataset in (pt, eta, phi) (in that order). If the dataset is projected down onto any one axis, the x-axis will automatically take on the correct range and values. Say we instead project down onto the 2D binning (pt, eta). Then the x-axis will be in units of bin index, but it is possible to label blocks of indices as corresponding to individual pT bins. This makes explicit on the plot the pT and eta dependence of the quantity being plotted. This is what is done by the fancy prebinned labels. 

The fancy prebinned labels have the following configuration options:

 - `fancy_prebinned_labels.enabled : bool` - a global on/off toggle for fancy prebinned labels 
 - `fancy_prebinned_labels.max_ndim : int` - the maximum number of dimensions in the histogram for the fancy labels to be created. For example, if the binning is 4-dimensional it might not be desireable (or interesting) to explicitely label the pT axis
 - `fancy_prebinned_labels.ticksize : int` - the size of the major ticks separating blocks of the outermost axis
 - `fancy_prebinned_labels.max_minor_ticks : int` - by default, the fancy prebinned labels enable minor ticks for every individual bin. If there are too many bins, howver, this just turns into a black stripe along the axis spine, which is ugly and not useful. If there would be more than `max_minor_ticks` minor ticks, we just use instead the default minor ticks from matplotlib. 
 - `fancy_prebinned_labels : fontsize : int` - the (maximum) font size of the axis block labels. Note that there is some automatic detection of overlapping labels (if they are too close together) which can reduce the font size used 
 - `fancy_prebinned_labels.check_label_overlap : bool` - whether to run the automatic detection and compensation for overlapping labels. This is obviously nice for automatic plot beauty, but can be slow as the size of text objects can not be guessed without rendering them. 
 - `fancy_prebinned_labels.min_fontsize : int` - the automatic label overlap resolution code iteratively decreases the label font size by 1 point and checks again for overlap. If there is still overlap after the font size is reduced to `min_fontsize` the code emits a warning, gives up, and rotates the axis block labels by 30 degrees. Note that this may cause collisions with the main axis label

### Axis labels

The `axis_labels` field contains a dictionary of lookups from internal variable keys to the text to be used on plot labels. These internal variable keys are the `variable.key` field, and can be column names from parquet or NANO files, histogram axis names, or the compositions produced by the various `Variable` classes. 

Note that the user `axis_labels` dict OVERWRITES the default dict, rather than simply updating its values.

The matching between `variable.key` fields and keys in the `axis_labels` dictionary allows the `*` wildcard to refer to any number of alphanumeric characters (but importantly not dots, underscores, parentheses, math symbols, etc). All other special characters are treated as literals (ie not as wildcards). 

For NanoAOD datasets, the generic structure for a column name is `collection.variable` (for example `Jet.pt`, `Muon.pt`, `FatJet.pt`, ...). To avoid having to write label rules for every possible collection name independently, variable names containing `.` characters have some additional matching logic:

First the code attempts the match including the collection names. If this is not successful, the collection names are stripped out of the variable key, and matching is attempted again. For example, say we have the following `axis_labels` dict:

```json
{
    "pt" : "$p_T$ [GeV]$",
    "*Jet.pt" : "Jet $p_T$ [GeV]"
}
```

Then the variables `FatJet.pt` and `Jet.pt` will both receive the labels `Jet $p_T$ [GeV]`. The variable `Muon.pt` will fail the first phase of matching, but with the collection name stripped will match with the first entry and recieve the label `$p_T$ [GeV]`. 

In any case, if a variable key does not match with any key in the `axis_labels` dict, then the label is just the variable key string.

### Automatic logarithmic x-axes

Passing `None` as the logx parameter to `plot_histogram` tells the code to automatically decide whether the x-axis should be logarithmic. This is controlled by the `auto_logx_patterns : List[str]` option. The logic here is simple: if the `key` for the variable being plotted matches (with the same wildcard rules as the `axis_labels` dict) any entry in the `auto_logx_patterns` list, then the x-axis will be logarithmic. 