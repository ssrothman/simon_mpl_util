import hist
from simon_mpl_util.plotting.typing.Protocols import  BinningKind

from abc import ABC, abstractmethod

class BinningBase(ABC):
    @property
    @abstractmethod
    def kind(self) -> BinningKind:
        raise NotImplementedError()

    @property
    @abstractmethod
    def has_custom_labels(self) -> bool:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def label_lookup(self) -> dict[str, str]:
        raise NotImplementedError()