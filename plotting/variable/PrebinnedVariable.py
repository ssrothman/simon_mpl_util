from .VariableBase import VariableBase
from typing import override

class PrebinnedVariable(VariableBase):
    def __init__(self):
        pass #stateless
    
    @property
    @override
    def _natural_centerline(self):
        return None
    
    @property 
    @override
    def columns(self):
        return []
    
    @property
    @override
    def prebinned(self) -> bool:
        return True
    
    @override
    def evaluate(self, dataset, cut):
        return cut.evaluate(dataset)

    @property 
    @override
    def key(self):
        return "PREBINNED"

    @override    
    def set_collection_name(self, collection_name):
        raise ValueError("PrebinnedVariable does not support set_collection_name")
