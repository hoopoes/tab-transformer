from abc import ABC, abstractmethod


class BaseMachiine(ABC):
    @property
    @abstractmethod
    def queries(self):
        pass

    @abstractmethod
    def load_dataset(self):
        pass
