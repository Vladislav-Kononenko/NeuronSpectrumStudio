from abc import ABC, abstractmethod
from typing import Tuple, Sequence

class EEGSource(ABC):
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def read_sample(self) -> Tuple[Sequence[float], float]:
        pass

    @abstractmethod
    def get_channel_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_sampling_rate(self) -> float:
        pass