from simon_mpl_util.plotting.config import lookup_axis_label

from typing import List, Protocol, Any, Union, Sequence, Tuple
from abc import ABC, abstractmethod

class VariableBase(ABC):
    '''
    Base class for `Variable`s, implementing basic common functionality
    '''
    @property
    @abstractmethod
    def _natural_centerline(self):
        raise NotImplementedError("You forgot to implement _natural_centerline!")

    @property
    @abstractmethod
    def prebinned(self) -> bool:
        raise NotImplementedError("You forgot to implement prebinned!")

    @property
    @abstractmethod
    def columns(self):
        raise NotImplementedError("You forgot to implement columns!")

    @abstractmethod
    def evaluate(self, dataset, cut) -> Any:
        raise NotImplementedError("You forgot to implement evaluate!")

    @property
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError("You forgot to implement key!")

    @abstractmethod
    def set_collection_name(self, collection_name):
        raise NotImplementedError("You forgot to implement set_collection_name!")

    def override_centerline(self, centerline):
        self._centerline = centerline

    def clear_override_centerline(self):
        if hasattr(self, '_centerline'):
            del self._centerline

    @property
    def centerline(self):
        if hasattr(self, '_centerline'):
            return self._centerline
        else:
            return self._natural_centerline
    
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