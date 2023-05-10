from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

Event = Tuple[float, float]


@dataclass
class Clip:
    data: np.array
    sample_rate: int
    label: str
    events: List[Event]


@dataclass
class WavFileRef:
    file_ref: str
    label: str
