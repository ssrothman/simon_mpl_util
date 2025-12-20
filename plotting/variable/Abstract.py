from simon_mpl_util.plotting.util.config import lookup_axis_label

class AbstractVariable:
    def override_centerline(self, centerline):
        self._centerline = centerline

    @property
    def _natural_centerline(self):
        raise NotImplementedError()
    
    @property
    def centerline(self):
        if hasattr(self, '_centerline'):
            return self._centerline
        else:
            return self._natural_centerline

    @property
    def prebinned(self) -> bool:
        raise NotImplementedError()

    @property
    def columns(self):
        raise NotImplementedError()

    def evaluate(self, dataset):
        raise NotImplementedError()

    @property
    def key(self):
        raise NotImplementedError()
    
    def override_label(self, label):
        self._label = label

    def clear_override_label(self):
        if hasattr(self, '_label'):
            del self._label

    @property
    def label(self):
        if hasattr(self, '_label') and self._label is not None:
            return self._label
        else:
            return lookup_axis_label(self.key)

    def __eq__(self, other):
        raise NotImplementedError()

    def set_collection_name(self, collection_name):
        raise NotImplementedError()

