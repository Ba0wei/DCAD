"""Plain DCAD uses the full EM-DCAD trainer with uniform masking."""
from dataclasses import dataclass
from .em_dcad import TrainConfig as EMConfig, train

@dataclass
class TrainConfig(EMConfig):
    adaptive_weight: float = 0.0

DCADConfig = TrainConfig
