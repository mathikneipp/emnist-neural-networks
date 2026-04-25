from abc import ABC, abstractmethod

class Loss (ABC):
    
    @abstractmethod
    @staticmethod
    def fn(y, y_pred):
        pass
    
    @abstractmethod
    @staticmethod
    def grad(y, y_pred):
        pass
    
