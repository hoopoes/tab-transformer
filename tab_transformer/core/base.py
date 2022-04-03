from abc import ABC, abstractmethod


class BaseMachine(ABC):
    @property
    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def load_data(self):
        pass
