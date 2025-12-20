from simon_mpl_util.util.AribtraryBinning import ArbitraryBinning

class AbstractCut:
    @property
    def columns(self):
        raise NotImplementedError()

    def evaluate(self, dataset):
        raise NotImplementedError()

    @property
    def key(self):
        raise NotImplementedError()

    @property
    def plottext(self):
        if hasattr(self, "_plottext"):
            return self._plottext
        else:
            return self._auto_plottext()

    def _auto_plottext(self):
        raise NotImplementedError()

    def override_plottext(self, plottext):
        self._plottext = plottext

    def clear_override_plottext(self):
        del self._plottext

    #equality operator
    def __eq__(self, other):
        #error message says what subclass raised the error
        raise NotImplementedError("Equality operator not implemented for subclass %s"%(type(self).__name__))
    
    def set_collection_name(self, collection_name):
        raise NotImplementedError()

class PrebinnedOperation(AbstractCut):
    @property
    def columns(self):
        return []    
    
    def resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        if not hasattr(self, '_resulting_binning'):
            self._resulting_binning = self._compute_resulting_binning(binning)

        return self._resulting_binning
        
    def _compute_resulting_binning(self, binning : ArbitraryBinning) -> ArbitraryBinning:
        raise NotImplementedError("PrebinnedOperation subclasses must implement compute_resulting_binning method")
    
