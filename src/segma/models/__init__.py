from segma.models.pyannet import PyanNet, PyanNetSlim

from .whisper.hydra import HydraWhisper
from .whisper.surgical import SurgicalWhisper
from .whisper.whiserimax import WhisperiMax
from .whisper.whisperidou import Whisperidou

Models = {
    "whisperidou": Whisperidou,
    "whisperimax": WhisperiMax,
    "pyannet": PyanNet,
    "pyannet_slim": PyanNetSlim,
    "surgical_whisper": SurgicalWhisper,
    "hydra_whisper": HydraWhisper,
}


__all__ = [
    "Whisperidou",
    "WhisperiMax",
    "PyanNet",
    "PyanNetSlim",
    "SurgicalWhisper",
    "HydraWhisper",
    "Models",
]
